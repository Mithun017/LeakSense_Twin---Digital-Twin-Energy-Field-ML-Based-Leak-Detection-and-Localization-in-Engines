# LeakSense Twin - Working Run Commands (CMD)

### 🚀 Direct Start (Full System)
If you are already in the `leak_sense_twin` folder, just run:
```cmd
python main_leak_detection_system.py
```

Otherwise, use this full command from anywhere:
```cmd
cd /d "C:\Users\MITHUN\Desktop\STUDIES\PROJECT\60.LeakSense_Twin - Digital Twin & Energy Field & ML Based Leak Detection and Localization in Engines\Development\Testing\Claude\leak_sense_twin" && python main_leak_detection_system.py
```

### 📊 Generate Data Only
```cmd
cd /d "C:\Users\MITHUN\Desktop\STUDIES\PROJECT\60.LeakSense_Twin - Digital Twin & Energy Field & ML Based Leak Detection and Localization in Engines\Development\Testing\Claude\leak_sense_twin" && python -c "from data_generation.synthetic_data_generator import SyntheticDataGenerator; gen = SyntheticDataGenerator(); data = gen.generate_complete_dataset(100); print(data.head())"
```

### 🩹 Run Evaluation Patch
```cmd
cd /d "C:\Users\MITHUN\Desktop\STUDIES\PROJECT\60.LeakSense_Twin - Digital Twin & Energy Field & ML Based Leak Detection and Localization in Engines\Development\Testing\Claude" && python replace_eval.py
```

### 📦 Install Dependencies
```cmd
cd /d "C:\Users\MITHUN\Desktop\STUDIES\PROJECT\60.LeakSense_Twin - Digital Twin & Energy Field & ML Based Leak Detection and Localization in Engines\Development\Testing\Claude\leak_sense_twin" && pip install -r requirements.txt
```