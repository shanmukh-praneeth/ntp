import subprocess
import os
import xml.etree.ElementTree as ET

def run_simulation_for_config(iat, qos):
    # -------------------------------
    # Step 0: Update Configuration.netsim (if needed)
    # -------------------------------
    config_path = r"C:\Users\admin\Documents\NetSim\Workspaces\workspace\IMT2022542_dt\configuration.netsim"
    tree = ET.parse(config_path)
    root = tree.getroot()

    # Example: if you still want to touch APPLICATION node but not change IAT/QoS
    for app in root.iter('APPLICATION'):
        if app.attrib.get('NAME') == 'App1_INTERACTIVE_GAMING':
            app.set('QOS', qos)
            for child in app:
                if child.tag == 'DL_INTER_ARRIVAL_TIME':
                    child.set('VALUE', str(iat))
                    break
            break

    tree.write(config_path)
    print(f"\n🛠️ Updated Configuration.netsim")

    # -------------------------------
    # Step 1: Run NetSim Simulation
    # -------------------------------
    netsim_exe = r"C:\Program Files\NetSim\Standard_v14_3\bin\bin_x64\NetSimcore.exe"
    netsim_apppath = r"C:\Program Files\NetSim\Standard_v14_3\bin\bin_x64"
    netsim_iopath = r"C:\Users\admin\Documents\NetSim\Workspaces\workspace\IMT2022542_dt"

    print("🚀 Running NetSim simulation...")
    subprocess.run([
        netsim_exe,
        "-apppath", netsim_apppath,
        "-iopath", netsim_iopath
    ], check=True)
    print("✅ NetSim simulation completed.")

    # -------------------------------
    # Step 2: Run ThroughputCalculator.exe
    # -------------------------------
    calculator_dir = r"C:\Program Files\NetSim\Standard_v14_3\Docs\Advanced_PlotScripts\Application_Packet_Log"
    calculator_exe = os.path.join(calculator_dir, "ThroughputCalculator.exe")
    app_log_path = os.path.join(netsim_iopath, "log", "Application_Packet_Log.csv")
    interval = "50"

    print("📈 Running ThroughputCalculator...")
    subprocess.run([calculator_exe, app_log_path, interval], check=True)
    print("✅ Throughput calculation completed.")

    # -------------------------------
    # Step 3: Run Data Split & Violation Scripts
    # -------------------------------

    scripts_dir = r"C:\Users\admin\Desktop\IMT2022542_dt_codes"

    scripts_to_run = [
        "split_data.py",
        "dt.py"
    ]

    for script in scripts_to_run:
        script_path = os.path.join(scripts_dir, script)
        print(f"🐍 Running {script}...")
        if script.endswith(".py"):
            subprocess.run(["python", script_path], check=True)
        print(f"✅ {script} completed.")

    print(f"🎯 Completed evaluation")

#if __name__ == "__main__":
#    run_simulation_for_config()