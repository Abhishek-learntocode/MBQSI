import os
import subprocess
import sys
import time

# --- CONFIGURATION ---
BASE_DIR = r"C:\Users\Abhishek\IIITH\IITH\INDEPENDENT_STUDY\PROJECT"
SCRIPTS_DIR = os.path.join(BASE_DIR, "PHASE6", "scripts")

# The Sequence of Scientific Execution
PIPELINE_STAGES = [
    {
        "name": "Standardization Gatekeeper",
        "script": "generate_ecosystem_surfaces.py",
        "desc": "Verifying raster integrity and generating the ecosystem manifest."
    },
    {
        "name": "Ecosystem Cube Engine",
        "script": "generate_ecosystem_cube.py",
        "desc": "Synthesizing richness, stability, and priority surfaces."
    },
    {
        "name": "Ecological Contribution Audit",
        "script": "generate_species_hotspot_contribution.py",
        "desc": "Quantifying species-level impacts and generating publication plots (Engineer/Sentinel)."
    },
    {
        "name": "QGIS Style Library",
        "script": "generate_qgis_styles.py",
        "desc": "Exporting dynamic semantic styles for cartography."
    },
    {
        "name": "Hotspot Atlas Preview",
        "script": "generate_hotspot_atlas.py",
        "desc": "Generating final 4-panel visual diagnostic."
    }
]

def run_pipeline():
    print("="*60)
    print(" MBSQI PHASE 6: ECOSYSTEM INTELLIGENCE MASTER RUNNER ")
    print("="*60)
    start_time = time.time()

    for i, stage in enumerate(PIPELINE_STAGES, 1):
        print(f"\n[STAGE {i}/{len(PIPELINE_STAGES)}] {stage['name']}")
        print(f"  > {stage['desc']}")
        
        script_path = os.path.join(SCRIPTS_DIR, stage['script'])
        
        try:
            # Run the sub-script
            result = subprocess.run([sys.executable, script_path], check=True, capture_output=True, text=True)
            # Print the success tail of the output
            lines = result.stdout.strip().split('\n')
            for line in lines[-3:]: # Show last 3 lines of success
                print(f"    {line}")
            
        except subprocess.CalledProcessError as e:
            print(f"\n[CRITICAL FAILURE] Stage {i} failed!")
            print(f"Error Output:\n{e.stderr}")
            sys.exit(1)

    duration = (time.time() - start_time) / 60
    print("\n" + "="*60)
    print(f" [SUCCESS] FULL ECOSYSTEM PIPELINE COMPLETED ")
    print(f" Total Time: {duration:.2f} minutes")
    print(f" Deliverables ready in: PHASE6/ecosystem_indices/ and PHASE6/reports/")
    print("="*60)

if __name__ == "__main__":
    run_pipeline()
