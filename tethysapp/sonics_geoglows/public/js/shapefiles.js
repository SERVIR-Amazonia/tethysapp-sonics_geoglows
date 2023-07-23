shp_control =  `
    <div class="control-group">
        <label for="shpFile" class="label-control">Area geográfica:</label>
        <input class="form-control" type="file" id="shpFile" accept=".shp">
        <br>
        <div style="color:white"> El archivo shapefile debe estar proyectado en EPSG:4326 (Coordenadas geográficas) </div>
    </div>`;

function insert_shapefile(){
    $("#shpFile").on("change",  function(){
        // Lee el archivo desde la entrada de archivos
        var file = document.getElementById('shpFile').files[0];
        // Crea un objeto FileReader para leer el archivo
        var reader = new FileReader();
        reader.onload = function(e) {
            // Convierte el archivo shapefile a GeoJSON usando shpjs
            shp(e.target.result).then(function(geojson) {
                // Crea una capa de Leaflet con los datos del archivo GeoJSON
                if (typeof layerSHP !== 'undefined') {
                    map.removeLayer(layerSHP)
                }
                layerSHP = L.geoJSON(geojson, { style: { weight: 1 } }).addTo(map);
                map.fitBounds(layerSHP.getBounds());
            });
        };
      // Lee el archivo como una URL de datos
      reader.readAsDataURL(file);
    });
}

