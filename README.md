# RobotDog Guide


## Getting Started (dont use uv on Jetson with GPU)
1. Clone the repository:
   ```bash
   git clone https://github.com/Satya1998-debug/robotdog_guide.git
   ```
2. Navigate to the project directory:
   ```bash
    cd robotdog_guide
   ```
3. Intall uv library:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```  
4. Sync the uv dependencies:
   ```bash
   uv sync
   ```
   
## update requirements (need to used when only CPU is used, GPU in Jetson is not supported yet using uv)
- after every new installation of any new package run the following command to update the requirements.txt file
```bash
   uv pip list --format=freeze | cut -d'=' -f1 > requirements_unified.txt
```

## Jetson Setup
- default Python 3.8 is installed on Jetson, so need to install python 3.10 using uv

```
- set python3.10 as default python version