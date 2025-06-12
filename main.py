import streamlit as st
import geopandas as gpd
import pandas as pd
import folium
from streamlit_folium import st_folium
import tempfile
import os
from shapely.geometry import Point
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
st.config.set_option('server.maxUploadSize', 1024)

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

# Initialize session state variables
if 'current_point' not in st.session_state:
    st.session_state.current_point = 0
if 'filtered_gdf' not in st.session_state:
    st.session_state.filtered_gdf = None
if 'selected_crop' not in st.session_state:
    st.session_state.selected_crop = 'All'
if 'map_center' not in st.session_state:
    st.session_state.map_center = [10.5, 78.5]
if 'map_zoom_level' not in st.session_state:
    st.session_state.map_zoom_level = 18  # Start with higher zoom level
if 'gdf' not in st.session_state:
    st.session_state.gdf = None
if 'batch_size' not in st.session_state:
    st.session_state.batch_size = 10
if 'current_batch' not in st.session_state:
    st.session_state.current_batch = 0
if 'last_uploaded_files_names' not in st.session_state:
    st.session_state.last_uploaded_files_names = []

@st.cache_data(show_spinner="Loading data...")
def load_data(uploaded_files_list_arg):
    temp_dir = tempfile.mkdtemp()
    shp_file_path = None
    csv_file_path = None

    if not uploaded_files_list_arg:
        if os.path.exists(temp_dir):
            try: os.rmdir(temp_dir)
            except OSError: pass
        return None

    uploaded_extensions = {os.path.splitext(f.name)[1].lower() for f in uploaded_files_list_arg}
    
    # Check for SHP files
    shp_required_extensions = {'.shp', '.shx', '.dbf'}
    is_shp_upload = any(ext in uploaded_extensions for ext in shp_required_extensions)
    
    # Check for CSV files
    is_csv_upload = '.csv' in uploaded_extensions

    if is_shp_upload and is_csv_upload:
        st.warning("Please upload either a Shapefile or a CSV file, not both.")
        return None
    
    if is_shp_upload and not uploaded_extensions.issuperset(shp_required_extensions):
        missing = shp_required_extensions - uploaded_extensions
        st.warning(
            f"Missing essential shapefile components: {', '.join(missing)}. "
            "Please ensure .shp, .shx, and .dbf files are uploaded for shapefile."
        )
        return None

    try:
        for uploaded_file_obj in uploaded_files_list_arg:
            file_path = os.path.join(temp_dir, uploaded_file_obj.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file_obj.getbuffer())
            if uploaded_file_obj.name.lower().endswith('.shp'):
                shp_file_path = file_path
            elif uploaded_file_obj.name.lower().endswith('.csv'):
                csv_file_path = file_path
        
        gdf = None
        if shp_file_path:
            with fiona.Env(SHAPE_RESTORE_SHX='YES'):
                gdf = gpd.read_file(
                    shp_file_path,
                    engine='pyogrio',
                    use_arrow=True
                )
        elif csv_file_path:
            df = pd.read_csv(csv_file_path)
            # Assuming CSV has 'latitude' and 'longitude' columns
            if 'latitude' not in df.columns or 'longitude' not in df.columns:
                raise ValueError("CSV file must contain 'latitude' and 'longitude' columns.")
            
            # Convert DataFrame to GeoDataFrame
            gdf = gpd.GeoDataFrame(
                df,
                geometry=gpd.points_from_xy(df.longitude, df.latitude),
                crs="EPSG:4326"
            )
            # Ensure geometry column is explicitly set to Point objects and CRS is maintained
            gdf['geometry'] = gdf.apply(lambda row: Point(row['longitude'], row['latitude']), axis=1)
            gdf.set_crs("EPSG:4326", allow_override=True, inplace=True)
        else:
            raise ValueError("No supported file (.shp or .csv) found among the uploaded files.")

        if 'crop_name' not in gdf.columns:
            # If crop_name is missing, create a dummy column or raise an error
            # For now, let's create a dummy column
            gdf['crop_name'] = "Unknown Crop"
            st.warning("The uploaded file doesn't have a 'crop_name' column. Using 'Unknown Crop'.")

        if 'S_No' not in gdf.columns:
            gdf['S_No'] = range(1, len(gdf) + 1)
        if 'validation' not in gdf.columns:
            gdf['validation'] = "Not Validated"
        else:
            # Map existing boolean values to string representations
            gdf['validation'] = gdf['validation'].apply(
                lambda x: "Correct" if x is True else ("Incorrect" if x is False else x)
            ).astype(str)

        # Remove duplicate records based on latitude and longitude
        initial_len = len(gdf)
        # Extract latitude and longitude into temporary columns for deduplication
        gdf['temp_latitude'] = gdf.geometry.y
        gdf['temp_longitude'] = gdf.geometry.x
        gdf.drop_duplicates(subset=['temp_latitude', 'temp_longitude'], inplace=True)
        # Drop the temporary columns
        gdf.drop(columns=['temp_latitude', 'temp_longitude'], inplace=True)

        if len(gdf) < initial_len:
            st.warning(f"Removed {initial_len - len(gdf)} duplicate points based on coordinates.")

        return gdf
    except Exception as e:
        st.error(f"Error processing uploaded file: {str(e)}")
        return None
    finally:
        if os.path.exists(temp_dir):
            for item in os.listdir(temp_dir):
                try: os.remove(os.path.join(temp_dir, item))
                except OSError: pass
            try: os.rmdir(temp_dir)
            except OSError: pass

def create_buffer_bounds(point, distance_meters=30):
    transformer_to_mercator = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    transformer_to_wgs84 = pyproj.Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    mx, my = transformer_to_mercator.transform(point.x, point.y)
    buffer = distance_meters
    min_mx, min_my = mx - buffer, my - buffer
    max_mx, max_my = mx + buffer, my + buffer
    lon_min, lat_min = transformer_to_wgs84.transform(min_mx, min_my)
    lon_max, lat_max = transformer_to_wgs84.transform(max_mx, max_my)
    return [[lat_min, lon_min], [lat_max, lon_max]]

def get_current_batch_points(gdf, batch_size=10):
    if gdf is None or gdf.empty:
        return pd.DataFrame()
    start_idx = st.session_state.current_batch * batch_size
    end_idx = start_idx + batch_size
    return gdf.iloc[start_idx:end_idx]

def update_filtered_data_and_reset_nav():
    if st.session_state.gdf is not None:
        if st.session_state.selected_crop != 'All':
            st.session_state.filtered_gdf = st.session_state.gdf[
                st.session_state.gdf['crop_name'] == st.session_state.selected_crop
            ].copy()
        else:
            st.session_state.filtered_gdf = st.session_state.gdf.copy()
        
        st.session_state.current_batch = 0
        st.session_state.current_point = 0
        # Auto-zoom to first point when filter changes
        if not st.session_state.filtered_gdf.empty:
            batch_points_df = get_current_batch_points(st.session_state.filtered_gdf, st.session_state.batch_size)
            if not batch_points_df.empty:
                buffer_bounds = create_buffer_bounds(batch_points_df.iloc[0].geometry, 30)
                st.session_state.map_center = [
                    (buffer_bounds[0][0] + buffer_bounds[1][0]) / 2,
                    (buffer_bounds[0][1] + buffer_bounds[1][1]) / 2
                ]
                st.session_state.map_zoom_level = 18

def zoom_to_current_point():
    if st.session_state.filtered_gdf is not None and not st.session_state.filtered_gdf.empty:
        batch_points_df = get_current_batch_points(st.session_state.filtered_gdf, st.session_state.batch_size)
        if not batch_points_df.empty and st.session_state.current_point < len(batch_points_df):
            current_point = batch_points_df.iloc[st.session_state.current_point]
            buffer_bounds = create_buffer_bounds(current_point.geometry, 30)
            st.session_state.map_center = [
                (buffer_bounds[0][0] + buffer_bounds[1][0]) / 2,
                (buffer_bounds[0][1] + buffer_bounds[1][1]) / 2
            ]
            st.session_state.map_zoom_level = 18 # Default zoom level for navigation

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
            st.session_state.current_point = st.session_state.filtered_gdf.index.get_loc(global_idx) % batch_size
            
            zoom_to_current_point()
        except Exception as e:
            st.error(f"Error navigating to selected point: {e}")

# Main app layout
st.title("🌱 Ground Truth Validation")

# Create sidebar for controls
# with st.sidebar:
#     st.header("Data Controls")
    
#     # File uploader
#     uploaded_files_list_widget = st.file_uploader(
#         "Upload Shapefile (.shp, .dbf, .shx)", 
#         type=['shp', 'dbf', 'shx', 'prj'], 
#         accept_multiple_files=True,
#         key="shp_uploader_widget"
#     )
with st.sidebar:
    st.header("Data Controls")

    # Custom label with white color
    st.markdown("<span style='color:white'>Upload Data (.shp, .dbf, .shx, .csv)</span>", unsafe_allow_html=True)

    # File uploader with no label
    uploaded_files_list_widget = st.file_uploader(
        label="",  # No label since we're displaying our own
        type=['shp', 'dbf', 'shx', 'prj', 'csv'],
        accept_multiple_files=True,
        key="data_uploader_widget"
    )
    current_uploaded_files_names = sorted([f.name for f in uploaded_files_list_widget]) if uploaded_files_list_widget else []

    if uploaded_files_list_widget:
        if st.session_state.gdf is None or current_uploaded_files_names != st.session_state.last_uploaded_files_names:
            with st.spinner("Loading..."):
                st.session_state.gdf = load_data(uploaded_files_list_widget)
                if st.session_state.gdf is not None:
                    st.session_state.last_uploaded_files_names = current_uploaded_files_names
                    st.session_state.selected_crop = 'All'
                    update_filtered_data_and_reset_nav()
                    st.success("Data loaded!")
                else:
                    st.session_state.last_uploaded_files_names = []
    
    if st.session_state.gdf is not None:
        unique_crops = sorted(st.session_state.gdf['crop_name'].unique())
        options = ['All'] + list(unique_crops)
        st.markdown("<span style='color:white'>Filter by Crop</span>", unsafe_allow_html=True)
        st.selectbox(
            label="",  # Hide built-in label
            options=options,
            key='selected_crop',
            on_change=update_filtered_data_and_reset_nav
        )
        
        if st.session_state.filtered_gdf is None and st.session_state.gdf is not None:
            update_filtered_data_and_reset_nav()

    if st.session_state.gdf is not None and not st.session_state.gdf.empty:
        csv_data = st.session_state.gdf.drop(columns=['geometry']).to_csv(index=False).encode('utf-8')
        # st.markdown("<span style='color:white'>Export Validated Data as CSV</span>", unsafe_allow_html=True)
        # st.download_button(
        #     label="Download",
        #     data=csv_data,
        #     file_name=f"validated_ground_truth_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        #     mime="text/csv",
        #     help="Download the entire dataset with updated validation statuses as a CSV file."
        # )

        st.markdown("<span style='color:white'>Export Filtered Data as CSV</span>", unsafe_allow_html=True)
        unique_crops_for_download = sorted(st.session_state.gdf['crop_name'].unique())
        download_options = ['All'] + list(unique_crops_for_download)
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
            help=f"Download the {selected_crop_for_download} dataset with updated validation statuses as a CSV file."
        )

# Main content area - single column layout to minimize scrolling
if st.session_state.gdf is not None and st.session_state.filtered_gdf is not None:
    if not st.session_state.filtered_gdf.empty:
        total_batches = math.ceil(len(st.session_state.filtered_gdf) / st.session_state.batch_size)
        if total_batches == 0: total_batches = 1
        
        batch_points_df = get_current_batch_points(st.session_state.filtered_gdf, st.session_state.batch_size)
        
        if not batch_points_df.empty:
            if st.session_state.current_point >= len(batch_points_df):
                st.session_state.current_point = len(batch_points_df) - 1 if len(batch_points_df) > 0 else 0

            current_global_idx = batch_points_df.index[st.session_state.current_point]
            current_point_details = batch_points_df.loc[current_global_idx]
            
            # Create two columns - left for batch info, right for main content
            col_left, col_right = st.columns([1, 3])
            
            with col_left:
                # Batch Points Overview in a sidebar box
                with st.container():
                    st.markdown("### Batch Points Overview")
                    st.dataframe(
                        batch_points_df[['S_No', 'crop_name', 'validation']]
                        .reset_index(drop=True)
                        .style.apply(lambda row: ['background-color: #3498db; color: white'] * len(row) 
                            if row.name == st.session_state.current_point 
                            else [''] * len(row), axis=1)
                        .set_properties(**{'font-size': '0.8rem'}),
                        height=150
                    )
                    
                    # Batch navigation controls
                    col_batch1, col_batch2 = st.columns(2)
                    with col_batch1:
                        st.markdown(f"<p style='color:black;'><b>Batch:</b> {st.session_state.current_batch + 1}/{total_batches}</p>", unsafe_allow_html=True)
                    with col_batch2:
                        st.markdown(f"<p style='color:black;'><b>Points:</b> {len(batch_points_df)}</p>", unsafe_allow_html=True)
                    
                    col_batch_nav1, col_batch_nav2 = st.columns(2)
                    with col_batch_nav1:
                        if st.button("◀ Prev Batch", disabled=st.session_state.current_batch == 0, help="Go to the previous batch", key="prev_batch", type="secondary", use_container_width=True):
                            st.session_state.current_batch -= 1
                            st.session_state.current_point = 0
                            zoom_to_current_point()
                            st.rerun()
                    with col_batch_nav2:
                        if st.button("Next Batch ▶", disabled=st.session_state.current_batch >= total_batches - 1, help="Go to the next batch", key="next_batch", type="secondary", use_container_width=True):
                            st.session_state.current_batch += 1
                            st.session_state.current_point = 0
                            zoom_to_current_point()
                            st.rerun()
                    col_nav1, col_nav2 = st.columns(2)
                    with col_nav1:
                        if st.button("◀ Prev Point", disabled=st.session_state.current_point == 0, help="Go to the previous point", key="prev_point", type="secondary", use_container_width=True):
                            st.session_state.current_point -= 1
                            zoom_to_current_point()
                            st.rerun()
                    with col_nav2:
                        if st.button("Next Point ▶", disabled=st.session_state.current_point >= len(batch_points_df) - 1, help="Go to the next point", key="next_point", type="secondary", use_container_width=True):
                            st.session_state.current_point += 1
                            zoom_to_current_point()
                            st.rerun()
                    col_validate1, col_validate2 = st.columns(2)
                    with col_validate1:
                        if st.button("✅ Correct", key=f"correct_btn_{current_global_idx}", use_container_width=True):
                            st.session_state.filtered_gdf.loc[current_global_idx, 'validation'] = "Correct"
                            st.session_state.gdf.loc[current_global_idx, 'validation'] = "Correct"
                            st.success("Point marked as Correct!")
                            st.rerun()
                    with col_validate2:
                        if st.button("❌ Incorrect", key=f"incorrect_btn_{current_global_idx}", use_container_width=True):
                            st.session_state.filtered_gdf.loc[current_global_idx, 'validation'] = "Incorrect"
                            st.session_state.gdf.loc[current_global_idx, 'validation'] = "Incorrect"
                            st.error("Point marked as Incorrect!")
                            st.rerun()
                    
                    st.markdown("---")
                    st.markdown("### Validation Summary")
                    total_points = len(st.session_state.filtered_gdf)
                    validated_points = st.session_state.filtered_gdf[
                        (st.session_state.filtered_gdf['validation'] == 'Correct') |
                        (st.session_state.filtered_gdf['validation'] == 'Incorrect')
                    ].shape[0]
                    
                    st.markdown(f"<p style='color:black;'><b>Validated:</b> {validated_points}/{total_points}</p>", unsafe_allow_html=True)

                    non_validated_points = st.session_state.filtered_gdf[
                        st.session_state.filtered_gdf['validation'] == 'Not Validated'
                    ]

                    if not non_validated_points.empty:
                        non_validated_batches = {}
                        for idx, row in non_validated_points.iterrows():
                            batch_num = math.floor(st.session_state.filtered_gdf.index.get_loc(idx) / st.session_state.batch_size) + 1
                            if batch_num not in non_validated_batches:
                                non_validated_batches[batch_num] = []
                            non_validated_batches[batch_num].append(f"S_No: {row['S_No']} (Crop: {row['crop_name']})")
                        
                        non_validated_options = []
                        for batch, points in sorted(non_validated_batches.items()):
                            non_validated_options.append(f"Batch {batch} ({len(points)} points)")
                            for point_info in points:
                                non_validated_options.append(f"  - {point_info}")
                        
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
                    # with st.container():
                    #     st.markdown(f"""
                    #     <div class="point-info-card" style="height:100px; overflow:auto; padding:7px; border:1px solid #ccc; border-radius:8px;">
                    #         <p style="margin-bottom: 2px;"><strong>Current Point:</strong> {st.session_state.current_point + 1}/{len(batch_points_df)}</p>
                    #         <p style="margin-bottom: 2px;"><strong>Crop:</strong> {current_point_details['crop_name']}</p>
                    #         <p style="margin-bottom: 2px;"><strong>ID:</strong> {current_point_details['S_No']}</p>
                    #         <p style="margin-bottom: 2px;"><strong>Status:</strong> {"✅ Validated" if current_point_details['validation'] else "❌ Not Validated"}</p>
                    #     </div>
                    #     """, unsafe_allow_html=True)
            
            with col_right:
                # Current point details in a compact card
                with st.container():
                        st.markdown(f"""
                        <div class="point-info-card" style="padding:5px; border:1px solid #ccc; border-radius:9px;">
                            <div style="display: flex; justify-content: space-between; margin-bottom: 2px;">
                                <p style="color:black; margin:0;"><strong>Current Point:</strong> {st.session_state.current_point + 1}/{len(batch_points_df)}</p>
                                <p style="color:black; margin:0;"><strong>ID:</strong> {current_point_details['S_No']}</p>
                            </div>
                            <div style="display: flex; justify-content: space-between; margin-bottom: 2px;">
                                <p style="color:black; margin:0;"><strong>Batch:</strong> {st.session_state.current_batch + 1}/{total_batches}</p>
                                <p style="color:black; margin:0;"><strong>Status:</strong> {current_point_details['validation']}</p>
                            </div>
                            <p style="margin-bottom: 2px; color:black;"><strong>Crop:</strong> {current_point_details['crop_name']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Map display
                if st.session_state.filtered_gdf is not None and not st.session_state.filtered_gdf.empty:
                    m = folium.Map(
                        location=st.session_state.map_center,
                        zoom_start=st.session_state.map_zoom_level,
                        tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
                        attr='Google Satellite',
                        control_scale=True
                    )
                    
                    batch_points_df_map = get_current_batch_points(st.session_state.filtered_gdf, st.session_state.batch_size)
                    
                    if not batch_points_df_map.empty:
                        map_current_point_idx = st.session_state.current_point
                        if map_current_point_idx >= len(batch_points_df_map):
                            map_current_point_idx = len(batch_points_df_map) - 1 if len(batch_points_df_map) > 0 else 0

                        current_global_idx_map = batch_points_df_map.index[map_current_point_idx]
                        
                        for idx, row in batch_points_df_map.iterrows():
                            icon_color = 'blue' if idx == current_global_idx_map else ('green' if row['validation'] == 'Correct' else ('red' if row['validation'] == 'Incorrect' else 'orange'))
                            icon_type = 'star' if idx == current_global_idx_map else ('check' if row['validation'] == 'Correct' else ('times' if row['validation'] == 'Incorrect' else 'question'))
                            
                            folium.Marker(
                                location=[row.geometry.y, row.geometry.x],
                                popup=f"""
                                <div style="width: 200px;">
                                    <h4>Point {row['S_No']}</h4>
                                    <p><strong>Crop:</strong> {row['crop_name']}</p>
                                    <p><strong>Status:</strong> {row['validation']}</p>
                                </div>
                                """,
                                icon=folium.Icon(color=icon_color, icon=icon_type, prefix='fa')
                            ).add_to(m)
                    
                    st_folium(
                        m, 
                        width=None,
                        height=450,
                        key="main_map_display",
                        returned_objects=[]
                    )
                
                # Point navigation and validation controls below the map
                # with st.container():
                #     st.markdown("---")
                    # col_nav1, col_nav2, col_nav3 = st.columns([1,1,2])
                    # with col_nav1:
                    #     if st.button("◀ Previous Point", disabled=st.session_state.current_point == 0):
                    #         st.session_state.current_point -= 1
                    #         zoom_to_current_point()
                    #         st.rerun()
                    # with col_nav2:
                    #     if st.button("Next Point ▶", disabled=st.session_state.current_point >= len(batch_points_df) - 1):
                    #         st.session_state.current_point += 1
                    #         zoom_to_current_point()
                    #         st.rerun()
                    # with col_nav3:
                    #     with st.form(key=f"validation_form_{current_global_idx}"):
                    #         current_status = st.checkbox(
                    #             "Mark as Validated", 
                    #             value=bool(current_point_details['validation']),
                    #             key=f"valid_checkbox_{current_global_idx}"
                    #         )
                    #         if st.form_submit_button("Save Validation"):
                    #             st.session_state.filtered_gdf.loc[current_global_idx, 'validation'] = current_status
                    #             st.session_state.gdf.loc[current_global_idx, 'validation'] = current_status
                    #             st.success("Saved!")
                    #             st.rerun()
        else:
            st.warning("No points in current batch.")
    else:
        st.warning("No data matches your filter criteria.")
else:
    st.markdown("<p style='color:black;'>Please upload a shapefile to begin</p>", unsafe_allow_html=True)