#include <iostream>
#include <vector>
#include <deque>
#include <unordered_map>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <string>
#include <fstream>

// -------------------------------------------------------------------
// LightGBM C API (simplified placeholder for actual inference)
// In production, you would link against lib_lightgbm.so and use:
//   #include <LightGBM/c_api.h>
// Here we assume a function that loads a model and predicts.
// -------------------------------------------------------------------
namespace lgb {
    // Dummy model handle
    struct BoosterHandle { void* ptr = nullptr; };
    
    // Load a trained LightGBM model from file
    BoosterHandle load_model(const std::string& path) {
        // BoosterHandle handle;
        // LGBM_BoosterCreateFromModelfile(path.c_str(), &num_iter, &handle.ptr);
        return BoosterHandle{};
    }
    
    // Predict for a single row of features (27 dimensions)
    double predict(BoosterHandle& booster, const std::vector<double>& features) {
        // double result;
        // LGBM_BoosterPredictForMat(booster.ptr, ...);
        // Placeholder: return weighted sum of features (for compilation only)
        double sum = 0.0;
        for (auto f : features) sum += f;
        return sum / features.size(); // Dummy
    }
}

// -------------------------------------------------------------------
// Configuration constants
// -------------------------------------------------------------------
constexpr int MAX_STOCKS = 256;
constexpr int WINDOW_SHORT = 5;
constexpr int WINDOW_MED  = 10;
constexpr int WINDOW_LONG  = 30;
constexpr int WINDOW_ZSCORE = 200;

// -------------------------------------------------------------------
// Ring buffer for efficient rolling window statistics
// -------------------------------------------------------------------
class RollingWindow {
public:
    explicit RollingWindow(int capacity) : capacity_(capacity), sum_(0.0) {
        buffer_.reserve(capacity);
    }
    
    void push(double value) {
        if (buffer_.size() == capacity_) {
            sum_ -= buffer_.front();
            buffer_.pop_front();
        }
        buffer_.push_back(value);
        sum_ += value;
    }
    
    double mean() const {
        if (buffer_.empty()) return 0.0;
        return sum_ / buffer_.size();
    }
    
    double stddev() const {
        if (buffer_.size() < 2) return 0.0;
        double m = mean();
        double var = 0.0;
        for (double v : buffer_) var += (v - m) * (v - m);
        return std::sqrt(var / (buffer_.size() - 1));  // sample std
    }
    
    bool full() const { return buffer_.size() == capacity_; }
    const std::deque<double>& data() const { return buffer_; }
    
private:
    int capacity_;
    std::deque<double> buffer_;
    double sum_;
};

// -------------------------------------------------------------------
// Per-stock state: maintains recent history and computes features
// -------------------------------------------------------------------
struct StockState {
    int stock_id;
    
    // Rolling windows for key raw signals
    RollingWindow book_imbalance_window{WINDOW_ZSCORE};
    RollingWindow imbalance_size_window{WINDOW_ZSCORE};
    RollingWindow wap_deviation_window{WINDOW_ZSCORE};
    RollingWindow mid_price_window{2};   // for returns
    
    // Rolling windows for velocity/acceleration (short, medium, long)
    RollingWindow imb_vel_5{WINDOW_SHORT};
    RollingWindow imb_vel_10{WINDOW_MED};
    RollingWindow imb_vel_30{WINDOW_LONG};
    RollingWindow imb_acc_std_5{WINDOW_SHORT};
    RollingWindow imb_acc_std_10{WINDOW_MED};
    RollingWindow imb_acc_std_30{WINDOW_LONG};
    RollingWindow spread_window_5{WINDOW_SHORT};
    RollingWindow spread_window_10{WINDOW_MED};
    RollingWindow spread_window_30{WINDOW_LONG};
    RollingWindow price_vol_5{WINDOW_SHORT};
    RollingWindow price_vol_10{WINDOW_MED};
    RollingWindow price_vol_30{WINDOW_LONG};
    
    // Previous values for diff calculations
    double prev_book_imbalance = 0.0;
    double prev_imb_velocity = 0.0;
    double prev_imbalance_size = 0.0;
    double prev_mid_price = 0.0;
    
    // Additional state
    double target_mean = 0.0;   // training set per-stock mean
    double target_std  = 1.0;   // training set per-stock std
};

// -------------------------------------------------------------------
// Main strategy class: computes features and returns prediction
// -------------------------------------------------------------------
class ClosingPricePredictor {
public:
    ClosingPricePredictor(const std::string& model_path) 
        : booster_(lgb::load_model(model_path)) {}
    
    // Call this on every new snapshot (order book + auction data)
    double on_snapshot(
        int stock_id,
        double seconds_in_bucket,
        double bid_price, double ask_price,
        double bid_size, double ask_size,
        double imbalance_size,
        double far_price, double near_price,
        double reference_price,
        double wap,
        double matched_size = 0.0
    ) {
        // Retrieve or create state for this stock
        auto& state = states_[stock_id];
        state.stock_id = stock_id;
        
        // ---------- Base signals ----------
        double mid_price = (bid_price + ask_price) / 2.0;
        double spread = ask_price - bid_price;
        double book_imbalance = (bid_size - ask_size) / (bid_size + ask_size + 1e-6);
        double wap_deviation = wap - mid_price;
        double auction_spread = far_price - near_price;
        double ref_price_deviation = (reference_price != 0) ? 
            (mid_price - reference_price) / reference_price : 0.0;
        
        // ---------- Update rolling windows ----------
        state.book_imbalance_window.push(book_imbalance);
        state.imbalance_size_window.push(imbalance_size);
        state.wap_deviation_window.push(wap_deviation);
        state.mid_price_window.push(mid_price);
        
        // ---------- Rolling z-score normalization ----------
        auto zscore = [](double x, const RollingWindow& w) -> double {
            double std = w.stddev();
            return (std > 1e-6) ? (x - w.mean()) / std : 0.0;
        };
        double book_imbalance_norm = zscore(book_imbalance, state.book_imbalance_window);
        double imbalance_size_norm = zscore(imbalance_size, state.imbalance_size_window);
        double wap_deviation_norm = zscore(wap_deviation, state.wap_deviation_window);
        
        // ---------- Velocity & acceleration ----------
        double imb_velocity = book_imbalance - state.prev_book_imbalance;
        double imb_acceleration = imb_velocity - state.prev_imb_velocity;
        double imbalance_velocity = imbalance_size - state.prev_imbalance_size;
        
        // Update rolling windows for derived signals
        state.imb_vel_5.push(imb_velocity);
        state.imb_vel_10.push(imb_velocity);
        state.imb_vel_30.push(imb_velocity);
        state.imb_acc_std_5.push(imb_acceleration);
        state.imb_acc_std_10.push(imb_acceleration);
        state.imb_acc_std_30.push(imb_acceleration);
        
        double spread_mean_5 = state.spread_window_5.mean();
        double spread_mean_10 = state.spread_window_10.mean();
        double spread_mean_30 = state.spread_window_30.mean();
        state.spread_window_5.push(spread);
        state.spread_window_10.push(spread);
        state.spread_window_30.push(spread);
        
        // Price volatility (using simple return)
        double price_return = 0.0;
        if (state.prev_mid_price > 1e-6) {
            price_return = (mid_price - state.prev_mid_price) / state.prev_mid_price;
        }
        state.price_vol_5.push(std::abs(price_return));
        state.price_vol_10.push(std::abs(price_return));
        state.price_vol_30.push(std::abs(price_return));
        
        // ---------- Additional features ----------
        double imb_vel_mean_5 = state.imb_vel_5.mean();
        double imb_vel_mean_10 = state.imb_vel_10.mean();
        double imb_vel_mean_30 = state.imb_vel_30.mean();
        double imb_acc_std_5 = state.imb_acc_std_5.stddev();
        double imb_acc_std_10 = state.imb_acc_std_10.stddev();
        double imb_acc_std_30 = state.imb_acc_std_30.stddev();
        
        double price_vol_5 = state.price_vol_5.mean();
        double price_vol_10 = state.price_vol_10.mean();
        double price_vol_30 = state.price_vol_30.mean();
        
        double imb_vs_mean_5 = book_imbalance - state.book_imbalance_window.mean(); // simplified
        double imb_vs_mean_30 = book_imbalance - state.book_imbalance_window.mean();
        
        double auction_book_divergence = (imbalance_size_norm - book_imbalance_norm) /
            (std::abs(imbalance_size_norm) + std::abs(book_imbalance_norm) + 1e-6);
        
        // ---------- Assemble feature vector (must match training order) ----------
        std::vector<double> features = {
            seconds_in_bucket,
            mid_price, spread, book_imbalance, wap_deviation,
            auction_spread, ref_price_deviation,
            book_imbalance_norm, imbalance_size_norm, wap_deviation_norm,
            imb_velocity, imb_acceleration, imbalance_velocity,
            imb_vel_mean_5, imb_vel_mean_10, imb_vel_mean_30,
            imb_acc_std_5, imb_acc_std_10, imb_acc_std_30,
            spread_mean_5, spread_mean_10, spread_mean_30,
            price_vol_5, price_vol_10, price_vol_30,
            imb_vs_mean_5, imb_vs_mean_30,
            auction_book_divergence
        };
        
        // ---------- Model inference (standardized prediction) ----------
        double pred_std = lgb::predict(booster_, features);
        
        // De-standardize to original target scale
        double prediction = pred_std * state.target_std + state.target_mean;
        
        // Update previous values for next tick
        state.prev_book_imbalance = book_imbalance;
        state.prev_imb_velocity = imb_velocity;
        state.prev_imbalance_size = imbalance_size;
        state.prev_mid_price = mid_price;
        
        return prediction;
    }
    
    // Set per-stock normalization parameters (from training data)
    void set_stock_params(int stock_id, double mean, double std) {
        states_[stock_id].target_mean = mean;
        states_[stock_id].target_std = std;
    }
    
private:
    std::unordered_map<int, StockState> states_;
    lgb::BoosterHandle booster_;
};

// -------------------------------------------------------------------
// Example usage
// -------------------------------------------------------------------
int main() {
    // Load model and set stock statistics (values from training set)
    ClosingPricePredictor predictor("model.txt");
    predictor.set_stock_params(0, -0.023, 3.15);
    predictor.set_stock_params(1,  0.011, 2.80);
    // ... for all stocks
    
    // Simulate a stream of market data snapshots
    // Format: stock_id, sec_in_bucket, bid_p, ask_p, bid_s, ask_s, imb_size, far_p, near_p, ref_p, wap
    std::vector<std::vector<double>> mock_data = {
        {0, 500, 100.5, 100.6, 200, 150, 500, 100.7, 100.4, 100.55, 100.53},
        {0, 501, 100.6, 100.7, 180, 170, 600, 100.8, 100.5, 100.55, 100.62},
        // ...
    };
    
    for (const auto& row : mock_data) {
        double pred = predictor.on_snapshot(
            (int)row[0], row[1], row[2], row[3], row[4], row[5],
            row[6], row[7], row[8], row[9], row[10]
        );
        std::cout << "Stock " << row[0] << " at second " << row[1] 
                  << " predicted future move: " << pred << std::endl;
    }
    
    return 0;
}
