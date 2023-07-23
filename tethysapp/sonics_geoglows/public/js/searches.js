search_control =  `
        <div class="control-group">
            <label class="label-control" for="select-comid">COMID del tramo de río:</label>
            <select id="select-comid" multiple placeholder="Escriba el COMID del río de interés."></select>
            <br>
        </div>`;


function dynamic_search_boxes(){
//  Select box for ZOOM to stations and rivers
fetch("get-rivers")
    .then((response) => (layer = response.json()))
    .then((layer) => {
        // Format json as input of selectize
        est_layer = layer.features.map(item => item.properties);
        
        // Rendering the select box for comid
        $('#select-comid').selectize({
            maxItems: 1,
            options: est_layer,
            valueField:  'comid',
            labelField:  'comid',
            searchField: 'comid',
            create: false,
            onChange: function(value, isOnInitialize) {
                // Station item selected
                river_item = est_layer.filter(item => item.comid == value);
                // Remove marker if exists
                if (typeof ss_marker !== 'undefined') {
                    map.removeLayer(ss_marker)
                }
                // Create the layer Groups that contain the selected stations
                ss_marker = L.layerGroup();
                // Add marker to visualize the selected stations
                river_item.map(item => {
                    //L.marker([item.latitud, item.longitud]).addTo(ss_river)
                    L.circleMarker([item.latitude, item.longitude], {
                        radius : 7,
                        color  : '#AD2745',
                        opacity: 0.75,
                      }).addTo(ss_marker);
                });
                ss_marker.addTo(map);
                
                // Coordinates of selected stations
                lon_item = river_item.map(item => item.longitude);
                lat_item = river_item.map(item => item.latitude);
                // Bounds
                southWest = L.latLng(Math.min(...lat_item), Math.min(...lon_item));
                northEast = L.latLng(Math.max(...lat_item), Math.max(...lon_item));
                bounds = L.latLngBounds(southWest, northEast);
                // Fit the map
                map.fitBounds(bounds);
            }
        });
    });
}


