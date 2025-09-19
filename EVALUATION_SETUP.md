# Evaluation and Neptune Logging Setup

## Overview
The training script now includes periodic evaluation of the policy and comprehensive logging with Neptune AI.

## Setup

### 1. Install Dependencies
```bash
pip install neptune python-dotenv
```

### 2. Configure Neptune
Create a `.env` file in the CombustionRL directory:
```bash
# Neptune AI Configuration
NEPTUNE_API_TOKEN=your_neptune_api_token_here
NEPTUNE_PROJECT=your_username/your_project_name
```

### 3. Get Neptune Credentials
1. Sign up at [neptune.ai](https://neptune.ai)
2. Create a new project
3. Get your API token from the settings
4. Update the `.env` file with your credentials

## New Features

### Evaluation
- **Fixed test conditions**: Uses 8 predefined combinations of temperature, pressure, and phi
- **Deterministic evaluation**: Uses argmax of policy output (no exploration)
- **Periodic evaluation**: Runs every N updates (default: 10)
- **Comprehensive metrics**: Reward, CPU time, violation rate, action distribution
- **Episode limits**: Maximum steps per evaluation episode (default: 1000)
- **Consistent tracking**: Same conditions every evaluation for improvement tracking

### Neptune Logging
- **Training metrics**: Episode rewards, losses, KL divergence, etc.
- **Evaluation metrics**: Separate logging for evaluation runs
- **Hyperparameters**: All training configuration logged
- **Artifacts**: Model checkpoints, logs, and final models uploaded
- **Real-time monitoring**: View progress in Neptune dashboard

## Usage

### Basic Training with Evaluation
```bash
python train_ppo_lstm.py --eval_interval 10
```

### Custom Evaluation Settings
```bash
python train_ppo_lstm.py \
    --eval_interval 5 \
    --max_eval_steps 2000 \
    --eval_time 2e-2 \
    --eval_temperatures 700 900 1100 1300 \
    --eval_pressures 0.5 2.0 5.0 10.0 \
    --eval_phis 0.6 0.8 1.0 1.2 1.4
```

### Without Neptune Logging
If you don't set up Neptune credentials, the script will continue without logging to Neptune.

## Output Files

### Training Logs
- `train_log.csv`: Training metrics per update
- `eval_log.csv`: Overall evaluation metrics per evaluation run
- `eval_conditions_log.csv`: Condition-specific evaluation metrics

### Neptune Dashboard
- Real-time training curves
- Evaluation performance over time
- Action distribution analysis
- Model artifacts and checkpoints

## Evaluation Metrics

- **Mean Reward**: Average episode reward
- **CPU Time**: Average computation time per step
- **Violation Rate**: Fraction of steps exceeding error tolerance
- **Action Distribution**: Ratio of CVODE vs QSS solver usage
- **Episode Length**: Average episode duration

## Fixed Test Conditions

The evaluation generates all combinations of the provided temperature, pressure, and phi values.

**Default conditions** (3×3×3 = 27 combinations):
- **Temperatures**: [800, 1000, 1200] K
- **Pressures**: [1.0, 3.0, 5.0] bar  
- **Phi values**: [0.8, 1.0, 1.2]

**Example**: With the default values, you get 27 test conditions like:
- T800_P1.0_Phi0.80
- T800_P1.0_Phi1.00
- T800_P1.0_Phi1.20
- T800_P3.0_Phi0.80
- ... (and so on for all combinations)

This ensures consistent evaluation across training updates, enabling proper tracking of policy improvement.

## Notes

- Evaluation runs in deterministic mode (no exploration)
- Uses the same observation normalization as training
- LSTM hidden states are reset for each evaluation episode
- Fixed initial conditions enable consistent improvement tracking
- All test conditions use the same evaluation time duration
