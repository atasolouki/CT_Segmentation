# CT_Segmentation

This project provides tools and scripts for 3D CT scan segmentation, visualization, and analysis using Python. It demonstrates how to process medical volumetric data, generate masks, extract 3D meshes, and compute evaluation metrics.

## Features

- Download and load CT scan and segmentation mask data (NRRD format)
- Crop volumes using world coordinates
- Generate spherical and anatomical masks
- Extract 3D meshes from masks and export as `.ply`
- Visualize meshes and slices with `vedo` and `matplotlib`
- Compute Dice score, Hausdorff distance, and volume for segmentation evaluation

## Requirements

- Python 3.8+
- See `requirements.txt` for all dependencies

## Setup

1. **Clone the repository:**
   ```
   git clone https://github.com/atasolouki/CT_Segmentation.git
   cd CT_Segmentation
   ```

2. **Create and activate a virtual environment:**
   ```
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```

4. **Start Jupyter Notebook or open `tasks.ipynb` in VS Code.**


## 🧪 Running the Tasks

All tasks are implemented in both:

- 📓 `tasks.ipynb` — for step-by-step interactive execution in Jupyter
- 🐍 `main.py` — the exact same logic in script form for terminal execution

---

### 🔽 What it does

Running the notebook or script will:

1. 📥 Download sample CT scan and mask data
2. ✂️ Crop the CT volumes based on provided bounding box
3. 🧱 Run marching cubes to generate 3D meshes
4. 👁️ Visualize middle slices in axial, coronal, and sagittal planes
5. 📊 Compute and display segmentation metrics


## Directory Structure

```
CT_Segmentation/
│
├── data/                # Downloaded and generated data (ignored by git)
├── venv/                # Virtual environment (ignored by git)
├── tasks.ipynb          # Main notebook with all processing steps
├── requirements.txt     # Python dependencies
├── .gitignore
└── README.md
```

## Notes

- The `data/` and `venv/` folders are excluded from version control.
- All outputs (meshes, images) are saved in the `data/` directory.

## License

MIT License

## Acknowledgements

- Uses [vedo](https://vedo.embl.es/), [trimesh](https://trimsh.org/), [scikit-image](https://scikit-image.org/), and [nrrd](https://pynrrd.readthedocs.io/).