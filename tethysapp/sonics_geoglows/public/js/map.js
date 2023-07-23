// ------------------------------------------------------------------------------------------------------------ //
//                                              INITIALIZE THE MAP                                              //
// ------------------------------------------------------------------------------------------------------------ //

// Ajust the map to the window height
const height = $(window).height() - 50;
$("#map-container").height(height);


// Set the map container
var map = L.map("map-container", {
    zoomControl: false,
}).setView([-9.3, -76], 6);


// Add the base map
L.tileLayer("http://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 18,
    attribution:
        '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
}).addTo(map);


// Add the zoom control
L.control.zoom({ 
    position: "bottomright"
}).addTo(map);


// Add drainage network
fetch('https://geoserver.hydroshare.org/geoserver/HS-9b6a7f2197ec403895bacebdca4d0074/wfs?service=WFS&version=1.1.0&request=GetFeature&typeName=peru_geoglows_drainage&outputFormat=application/json', {
    method: 'GET',
    headers: {'Content-Type': 'application/json'}
})
.then(response => response.json())
.then(data => {
    rivers = L.geoJSON(data, {
        style: {
            weight: 1,
            color: "#4747C9",
            zIndex: 10000
        }
    }).addTo(map);
    rivers.on('click', showPanel)
    console.log("Esta ejecutado")
})
.catch(error => console.error('Error fetching data:', error));



// Add stations (Aqui nos quedamos)
fetch("get-alerts",{
    method: 'GET',
    headers: {'Content-Type': 'application/json'}
})
.then((response) => (layer = response.json()))
.then((layer) => {
    
        est_R002 = add_station_icon_R2(layer)
        est_R002.addTo(map);
        est_R002.on('click', showPanel)
        
        est_R005 = add_station_icon_R5(layer)
        est_R005.addTo(map);
        est_R005.on('click', showPanel)

        est_R010 = add_station_icon_R10(layer)
        est_R010.addTo(map);
        est_R010.on('click', showPanel)

}); 
