import streamlit as st
import geopandas as gpd
import pandas as pd
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium
import tempfile
import os
from shapely.geometry import Point, Polygon
import math
import pyproj
import fiona
from datetime import datetime

# Set page config with a more compact layout
st.set_page_config(
    layout="wide",
    page_title="Ground Truth Validator",
    page_icon="🌱"
)

# Increase upload size limit to 1GB
# This is a deprecated way. For Streamlit Cloud, configure in settings.
# For local, run with: streamlit run your_script.py --server.maxUploadSize 1024
# st.config.set_option('server.maxUploadSize', 1024)

# Custom CSS for improved UI and to minimize scrolling
st.markdown("""
<style>
/* Main container styling */
.stApp {
    background-color: #f5f7fa;
    padding-top: 1rem;
}

/* Compact layout adjustments */
[data-testid="stSidebar"] {
    background-color: #2c3e50 !important;
    color: white !important;
    padding: 1rem !important;
}

/* Header styling */
h1, h2, h3, h4, h5, h6 {
    color: #2c3e50 !important;
    margin-top: 0.5rem !important;
}

/* Sidebar header styling */
[data-testid="stSidebar"] h2 {
    color: white !important;
}

/* Button styling */
div[data-testid="stButton"] button {
    background-color: #3498db !important;
    color: white !important;
    border: none !important;
    border-radius: 5px !important;
    padding: 3px 6px !important;
    margin: 1px 0 !important;
    font-size: 0.75rem !important;
    transition: all 0.3s ease !important;
}

div[data-testid="stButton"] button:hover:not(:disabled) {
    background-color: #2980b9 !important;
}

div[data-testid="stButton"] button:disabled {
    background-color: #95a5a6 !important;
    opacity: 0.7 !important;
}

/* Compact form elements */
[data-testid="stVerticalBlock"] {
    gap: 0.2rem !important;
}

/* Smaller dataframes */
.dataframe {
    font-size: 0.85rem !important;
}

/* Remove extra padding in containers */
.stContainer {
    padding: 0 !important;
}

/* Custom card styling */
.custom-card {
    background-color: white;
    border-radius: 8px;
    padding: 10px;
    margin-bottom: 10px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    font-size: 0.9rem;
}

/* Make map container fit better */
.element-container:has(> [data-testid="stVerticalBlockBorderWrapper"] > div > .stDeckGlJsonChart) {
    height: calc(100vh - 200px) !important;
}

/* Sidebar dataframe styling */
.sidebar .dataframe {
    width: 100% !important;
}

/* Point info card styling */
.point-info-card {
    background-color: white;
    border-radius: 8px;
    padding: 10px;
    margin-bottom: 10px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    font-size: 0.9rem;
    border-left: 4px solid #3498db;
}
</style>
""", unsafe_allow_html=True)

# --- Session State Initialization ---
# Using complex objects in session state, so initialize carefully
if 'gdf' not in st.session_state:
    st.session_state.gdf = None
if 'filtered_gdf' not in st.session_state:
    st.session_state.filtered_gdf = None
if 'current_point_idx' not in st.session_state:
    st.session_state.current_point_idx = 0
if 'current_batch' not in st.session_state:
    st.session_state.current_batch = 0
if 'selected_crop' not in st.session_state:
    st.session_state.selected_crop = 'All'
if 'batch_size' not in st.session_state:
    st.session_state.batch_size = 10
if 'map_center' not in st.session_state:
    st.session_state.map_center = [10.5, 78.5]
if 'map_zoom' not in st.session_state:
    st.session_state.map_zoom = 18
if 'last_uploaded_files' not in st.session_state:
    st.session_state.last_uploaded_files = []
if 'map_data' not in st.session_state:
    st.session_state.map_data = None


# --- Core Logic Functions ---

@st.cache_data(show_spinner="Loading and processing data...")
def load_data(uploaded_files):
    """Loads data from SHP or CSV, de-duplicates, and prepares GeoDataFrame."""
    if not uploaded_files:
        return None

    temp_dir = tempfile.mkdtemp()
    shp_path, csv_path = None, None
    try:
        for f in uploaded_files:
            path = os.path.join(temp_dir, f.name)
            with open(path, "wb") as out:
                out.write(f.getbuffer())
            if f.name.lower().endswith('.shp'):
                shp_path = path
            elif f.name.lower().endswith('.csv'):
                csv_path = path

        if shp_path:
            gdf = gpd.read_file(shp_path)
        elif csv_path:
            df = pd.read_csv(csv_path)
            if 'latitude' not in df.columns or 'longitude' not in df.columns:
                st.error("CSV must contain 'latitude' and 'longitude' columns.")
                return None
            gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326")
        else:
            st.warning("Please upload a complete Shapefile or a CSV file.")
            return None

        if gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs("EPSG:4326")

        # Standardize columns
        if 'S_No' not in gdf.columns:
            gdf['S_No'] = range(1, len(gdf) + 1)
        if 'crop_name' not in gdf.columns:
            gdf['crop_name'] = "Unknown"
        if 'validation' not in gdf.columns:
            gdf['validation'] = "Not Validated"
        else:
            gdf['validation'] = gdf['validation'].apply(
                lambda x: "Correct" if x is True else ("Incorrect" if x is False else str(x))
            )

        # De-duplicate based on geometry
        initial_len = len(gdf)
        gdf = gdf.drop_duplicates(subset='geometry')
        if len(gdf) < initial_len:
            st.toast(f"Removed {initial_len - len(gdf)} duplicate points.")

        return gdf.reset_index(drop=True)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None
    finally:
        # Cleanup temp directory
        for item in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, item))
        os.rmdir(temp_dir)

def update_filtered_data():
    """Filters the main GDF based on selected crop and resets navigation."""
    if st.session_state.gdf is not None:
        crop = st.session_state.selected_crop
        if crop == 'All':
            st.session_state.filtered_gdf = st.session_state.gdf.copy()
        else:
            st.session_state.filtered_gdf = st.session_state.gdf[st.session_state.gdf['crop_name'] == crop].copy()
        
        # Reset navigation
        st.session_state.current_batch = 0
        st.session_state.current_point_idx = 0
        zoom_to_point()

def zoom_to_point(point_index=None):
    """Updates map center and zoom level to focus on a specific point."""
    if point_index is None:
        point_index = st.session_state.current_point_idx
    
    if st.session_state.filtered_gdf is not None and not st.session_state.filtered_gdf.empty:
        batch_df = get_current_batch_df()
        if not batch_df.empty and point_index < len(batch_df):
            point_geom = batch_df.iloc[point_index].geometry
            st.session_state.map_center = [point_geom.y, point_geom.x]
            st.session_state.map_zoom = 18

# --- Callback Functions for Performance ---
# These functions only modify session_state, preventing full reruns for simple actions.

def next_point():
    batch_df = get_current_batch_df()
    if st.session_state.current_point_idx < len(batch_df) - 1:
        st.session_state.current_point_idx += 1
        zoom_to_point()

def prev_point():
    if st.session_state.current_point_idx > 0:
        st.session_state.current_point_idx -= 1
        zoom_to_point()

def next_batch():
    total_batches = math.ceil(len(st.session_state.filtered_gdf) / st.session_state.batch_size)
    if st.session_state.current_batch < total_batches - 1:
        st.session_state.current_batch += 1
        st.session_state.current_point_idx = 0
        zoom_to_point()

def prev_batch():
    if st.session_state.current_batch > 0:
        st.session_state.current_batch -= 1
        st.session_state.current_point_idx = 0
        zoom_to_point()

def set_validation(status):
    """Sets validation status for the current point and moves to the next."""
    batch_df = get_current_batch_df()
    if not batch_df.empty:
        global_idx = batch_df.index[st.session_state.current_point_idx]
        st.session_state.gdf.loc[global_idx, 'validation'] = status
        st.session_state.filtered_gdf.loc[global_idx, 'validation'] = status
        st.toast(f"Point {st.session_state.gdf.loc[global_idx, 'S_No']} marked as {status}!")
        next_point() # Auto-navigate to the next point

def set_validation_by_s_no(s_no, status):
    """Sets validation status for a point identified by S_No (from popup) and moves to the next."""
    # Find the index in the main GDF
    idx_list = st.session_state.gdf[st.session_state.gdf['S_No'] == s_no].index
    if not idx_list.empty:
        global_idx = idx_list[0]
        st.session_state.gdf.loc[global_idx, 'validation'] = status
        # Also update the filtered GDF if the point exists there
        if global_idx in st.session_state.filtered_gdf.index:
            st.session_state.filtered_gdf.loc[global_idx, 'validation'] = status
        st.toast(f"Point {s_no} marked as {status} via popup!")
        next_point() # Auto-navigate to the next point

def bulk_validate(indices, status):
    """Validates a list of points by their global indices."""
    st.session_state.gdf.loc[indices, 'validation'] = status
    # Find which of these indices are in the current filtered view and update them too
    filtered_indices_to_update = st.session_state.filtered_gdf.index.intersection(indices)
    st.session_state.filtered_gdf.loc[filtered_indices_to_update, 'validation'] = status
    st.success(f"Validated {len(indices)} points as {status}!")
    # Clear the drawing from the map state
    if st.session_state.map_data and 'all_drawings' in st.session_state.map_data:
        st.session_state.map_data['all_drawings'] = None

def on_non_validated_point_select():
    selected_option = st.session_state.non_validated_points_dropdown
    if "S_No:" in selected_option:
        try:
            s_no_str = selected_option.split("S_No: ")[1].split(" ")[0]
            s_no = int(s_no_str)
            
            # Find the global index of the selected point
            global_idx = st.session_state.filtered_gdf[st.session_state.filtered_gdf['S_No'] == s_no].index[0]
            
            # Calculate the batch and point within the batch
            batch_size = st.session_state.batch_size
            st.session_state.current_batch = math.floor(st.session_state.filtered_gdf.index.get_loc(global_idx) / batch_size)
            st.session_state.current_point_idx = st.session_state.filtered_gdf.index.get_loc(global_idx) % batch_size
            
            zoom_to_point()
        except Exception as e:
            st.error(f"Error navigating to selected point: {e}")

@st.cache_data(show_spinner=False)
def get_non_validated_options_cached(_filtered_gdf, batch_size):
    if _filtered_gdf is None or _filtered_gdf.empty:
        return []

    non_validated_points = _filtered_gdf[
        _filtered_gdf['validation'] == 'Not Validated'
    ]

    if non_validated_points.empty:
        return []

    non_validated_batches = {}
    for idx, row in non_validated_points.iterrows():
        global_idx_in_filtered = _filtered_gdf.index.get_loc(idx)
        batch_num = math.floor(global_idx_in_filtered / batch_size) + 1
        if batch_num not in non_validated_batches:
            non_validated_batches[batch_num] = []
        non_validated_batches[batch_num].append(f"S_No: {row['S_No']} (Crop: {row['crop_name']})")
    
    non_validated_options = []
    for batch, points in sorted(non_validated_batches.items()):
        non_validated_options.append(f"Batch {batch} ({len(points)} points)")
        for point_info in points:
            non_validated_options.append(f"  - {point_info}")
    return non_validated_options

# --- Helper Functions ---
def get_current_batch_df():
    """Returns the GeoDataFrame for the current batch."""
    if st.session_state.filtered_gdf is None or st.session_state.filtered_gdf.empty:
        return gpd.GeoDataFrame()
    start = st.session_state.current_batch * st.session_state.batch_size
    end = start + st.session_state.batch_size
    return st.session_state.filtered_gdf.iloc[start:end]

# --- Main App ---
st.title("🌱 Ground Truth Validator")

# --- Sidebar ---
with st.sidebar:
    st.markdown("<h2 style='color:white;'>Data Controls</h2>", unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Upload Shapefile (.shp, .dbf, .shx) or CSV",
        type=['shp', 'dbf', 'shx', 'prj', 'csv'],
        accept_multiple_files=True,
        key="data_uploader"
    )

    # Load data only if new files are uploaded
    if uploaded_files and uploaded_files != st.session_state.last_uploaded_files:
        st.session_state.gdf = load_data(uploaded_files)
        st.session_state.last_uploaded_files = uploaded_files
        st.session_state.selected_crop = 'All' # Reset filter on new upload
        update_filtered_data()

    if st.session_state.gdf is not None:
        st.markdown("<span style='color:white'>Filter by Crop</span>", unsafe_allow_html=True)
        unique_crops = ['All'] + sorted(st.session_state.gdf['crop_name'].unique().tolist())
        st.selectbox(
            "Select Crop",
            options=unique_crops,
            key='selected_crop',
            on_change=update_filtered_data,
            label_visibility="collapsed"
        )

        st.markdown("<span style='color:white'>Export Filtered Data as CSV</span>", unsafe_allow_html=True)
        unique_crops_for_download = sorted(st.session_state.gdf['crop_name'].unique().tolist())
        download_options = ['All'] + unique_crops_for_download
        selected_crop_for_download = st.selectbox(
            label="Select Crop for Download",
            options=download_options,
            key='selected_crop_for_download',
            label_visibility="collapsed" # Hide built-in label
        )

        if selected_crop_for_download == 'All':
            filtered_download_gdf = st.session_state.gdf.copy()
        else:
            filtered_download_gdf = st.session_state.gdf[
                st.session_state.gdf['crop_name'] == selected_crop_for_download
            ].copy()

        filtered_csv_data = filtered_download_gdf.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=f"Download {selected_crop_for_download} Data",
            data=filtered_csv_data,
            file_name=f"validated_ground_truth_{selected_crop_for_download}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            key="download_filtered_data", # Added unique key
            help=f"Download the {selected_crop_for_download} dataset with updated validation statuses as a CSV file." # Corrected typo
        )

# --- Handle Popup Button Clicks (Improvement 2) ---
# This logic runs at the top to process the action before the page renders.
if 'action' in st.query_params:
    action = st.query_params['action']
    status = st.query_params.get('status')
    s_no = int(st.query_params.get('s_no'))
    
    if action == 'validate' and s_no and status:
        set_validation_by_s_no(s_no, status)
        # Clear query params to prevent re-triggering on refresh
        st.query_params.clear()


# --- Main Content Area ---
if st.session_state.filtered_gdf is None or st.session_state.filtered_gdf.empty:
    st.markdown("<p style='color:black;'>Please upload data to begin or select a crop with available points.</p>", unsafe_allow_html=True)
else:
    batch_df = get_current_batch_df()
    total_points_in_filter = len(st.session_state.filtered_gdf)
    total_batches = math.ceil(total_points_in_filter / st.session_state.batch_size)

    if batch_df.empty:
        st.markdown("<p style='color:black;'>No points in the current filter or batch.</p>", unsafe_allow_html=True)
    else:
        # Ensure current_point_idx is valid
        if st.session_state.current_point_idx >= len(batch_df):
            st.session_state.current_point_idx = 0
        
        current_point_details = batch_df.iloc[st.session_state.current_point_idx]
        
        col_left, col_right = st.columns([1, 3])

        with col_left:
            st.markdown("### Batch Overview")
            # Display batch points with highlight
            display_df = batch_df[['S_No', 'crop_name', 'validation']].reset_index(drop=True)
            st.dataframe(
                    display_df.style.apply(lambda row: ['background-color: #3498db; color: white'] * len(row) 
                        if row.name == st.session_state.current_point_idx else [''] * len(row), axis=1),
                height=150,
                use_container_width=True
            )
            
            # Batch Progress Indicator
            st.markdown(f"<p style='color:black;'>(<b>Batch:</b> {st.session_state.current_batch + 1} / {total_batches} | <b>Points:</b> {st.session_state.current_point_idx + 1} / {len(batch_df)})</p>", unsafe_allow_html=True)

            # --- Navigation and Validation Controls ---
            st.markdown("#### Navigation")
            
            b_col1, b_col2 = st.columns(2)
            b_col1.button("◀ Prev Batch", on_click=prev_batch, disabled=st.session_state.current_batch == 0, use_container_width=True)
            b_col2.button("Next Batch ▶", on_click=next_batch, disabled=st.session_state.current_batch >= total_batches - 1, use_container_width=True)

            p_col1, p_col2 = st.columns(2)
            p_col1.button("◀ Prev Point", on_click=prev_point, disabled=st.session_state.current_point_idx == 0, use_container_width=True)
            p_col2.button("Next Point ▶", on_click=next_point, disabled=st.session_state.current_point_idx >= len(batch_df) - 1, use_container_width=True)

            st.markdown("#### Validation")
            v_col1, v_col2 = st.columns(2)
            v_col1.button("✅Correct", on_click=set_validation, args=("Correct",), use_container_width=True)
            v_col2.button("❌wrong", on_click=set_validation, args=("Incorrect",), use_container_width=True)
            
            st.markdown("#### Validation Summary")
            validated_count = st.session_state.filtered_gdf['validation'].isin(['Correct', 'Incorrect']).sum()
            st.markdown(f"<p style='color:black;'><b>Validated:</b> {validated_count} / {total_points_in_filter}</p>", unsafe_allow_html=True)

            non_validated_options = get_non_validated_options_cached(
                st.session_state.filtered_gdf, st.session_state.batch_size
            )

            if non_validated_options:
                st.markdown("<span style='color:black'>Non-Validated Points (by Batch)</span>", unsafe_allow_html=True)
                st.selectbox(
                    label="",
                    options=non_validated_options,
                    key='non_validated_points_dropdown',
                    help="List of points that are not yet validated, grouped by batch.",
                    on_change=on_non_validated_point_select
                )
            else:
                st.markdown("<p style='color:black;'>All points in the current filter are validated!</p>", unsafe_allow_html=True)


        with col_right:
            # --- Current Point Info Card ---
            st.markdown(f"""
            <div class="point-info-card">
                <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                    <span style="color:black;"><strong>Point ID:</strong> {current_point_details['S_No']}</span>
                    <span style="color:black;"><strong>Status:</strong> {current_point_details['validation']}</span>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span style="color:black;"><strong>Batch:</strong> {st.session_state.current_batch + 1}/{total_batches}</span>
                    <span style="color:black;"><strong>Point in Batch:</strong> {st.session_state.current_point_idx + 1}/{len(batch_df)}</span>
                </div>
                <p style="margin-top: 5px; margin-bottom: 0;"><strong>Crop:</strong> {current_point_details['crop_name']}</p>
            </div>
            """, unsafe_allow_html=True)

            # --- Map Display ---
            st.markdown(f"<p style='color:black;'>Map will attempt to render with {len(st.session_state.filtered_gdf)} points.</p>", unsafe_allow_html=True)
            m = folium.Map(
                location=st.session_state.map_center,
                zoom_start=st.session_state.map_zoom,
                tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
                attr='Google Satellite',
                control_scale=True
            )

            # Add Draw plugin for bounding box (Improvement 1)
            Draw(
                export=True,
                filename='selection.geojson',
                position='topleft',
                draw_options={'rectangle': {'shapeOptions': {'color': '#0078A8'}}}
            ).add_to(m)

            # Add markers for the current batch
            for idx, row in batch_df.iterrows():
                is_current = (idx == current_point_details.name)
                
                if row['validation'] == 'Correct':
                    color, icon = 'green', 'check'
                elif row['validation'] == 'Incorrect':
                    color, icon = 'red', 'times'
                else:
                    color, icon = 'orange', 'question'
                
                if is_current:
                    color, icon = 'blue', 'star'

                # Improvement 2: Add validation buttons to popup
                popup_html = f"""
                <div style="width: 200px;">
                    <h4>Point {row['S_No']}</h4>
                    <p><strong>Crop:</strong> {row['crop_name']}</p>
                    <p><strong>Status:</strong> {row['validation']}</p>
                    <hr>
                    <a href="?action=validate&status=Correct&s_no={row['S_No']}" target="_self" style="background-color: #28a745; color: white; padding: 5px 10px; text-decoration: none; border-radius: 5px; margin-right: 5px;">✅ Correct</a>
                    <a href="?action=validate&status=Incorrect&s_no={row['S_No']}" target="_self" style="background-color: #dc3545; color: white; padding: 5px 10px; text-decoration: none; border-radius: 5px;">❌ Incorrect</a>
                </div>
                """
                
                folium.Marker(
                    location=[row.geometry.y, row.geometry.x],
                    popup=folium.Popup(popup_html),
                    icon=folium.Icon(color=color, icon=icon, prefix='fa')
                ).add_to(m)

            # Render the map
            map_output = st_folium(
                m,
                key="folium_map",
                width='100%',
                height=400,
                returned_objects=["all_drawings"]
            )

            # --- Bounding Box Validation Logic (Improvement 1) ---
            if map_output and map_output["all_drawings"]:
                st.markdown("### Bulk Validation")
                st.markdown("<p style='color:black;'>A bounding box was drawn. Use the buttons below to validate all points inside.</p>", unsafe_allow_html=True)
                
                # Get the last drawn shape
                last_drawing = map_output["all_drawings"][-1]
                coords = last_drawing['geometry']['coordinates'][0]
                drawn_polygon = Polygon(coords)

                # Find points within the polygon from the current batch only
                points_in_box = batch_df[batch_df.within(drawn_polygon)]
                
                if not points_in_box.empty:
                    st.markdown(f"<p style='color:black;'>Found <b>{len(points_in_box)}</b> points in the selected area.</p>", unsafe_allow_html=True)
                    st.dataframe(points_in_box[['S_No', 'crop_name', 'validation']], height=150)
                    
                    bb_col1, bb_col2 = st.columns(2)
                    bb_col1.button(
                        f"✅ Mark all {len(points_in_box)} as Correct",
                        on_click=bulk_validate,
                        args=(points_in_box.index, "Correct"),
                        use_container_width=True
                    )
                    bb_col2.button(
                        f"❌ Mark all {len(points_in_box)} as Incorrect",
                        on_click=bulk_validate,
                        args=(points_in_box.index, "Incorrect"),
                        use_container_width=True
                    )
                else:
                    st.markdown("<p style='color:black;'>No points found within the drawn bounding box.</p>", unsafe_allow_html=True)
