# RobotDog Guide



## Available MCP Tools


## Getting Started
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
   

## update requirements
- after every new installation of any new package run the following command to update the requirements.txt file
```bash
   uv pip list --format=freeze | cut -d'=' -f1 > requirements_unified.txt
```

### get the door dataset
- download the door dataset from https://github.com/MiguelARD/DoorDetect-Dataset.git
- save it inside door_dataset folder in object_detection directory

## Contributing
Contributions are welcome! Please open issues or submit pull requests for improvements.

## License
This project is licensed under the MIT License.
