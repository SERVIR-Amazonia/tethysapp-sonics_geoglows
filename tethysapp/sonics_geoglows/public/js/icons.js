// ------------------------------------------------------------------------------------------------------------ //
//                                     COLOR MARKER ACCORDING TO THE ALERT                                      //
// ------------------------------------------------------------------------------------------------------------ //

// Function to construct Icon Marker 
function IconMarker(type, rp) {
    const IconMarkerR = new L.Icon({
      iconUrl: `${server}/static/${app_name}/images/${type}/${rp}.svg`,
      iconSize: [16, 16],
      iconAnchor: [8, 8],
    });
    return IconMarkerR;
  }


// R2 ------------------------------------------------------------------------------------
const station_R002 = IconMarker("alert","2");
function station_icon_R2(feature, latlng) {
    return L.marker(latlng, { icon: station_R002 });
}
function add_station_icon_R2(layer){
    const st = L.geoJSON(layer.features.filter(item => item.properties.alert === "R2"), {
        pointToLayer: station_icon_R2,
    });
    return(st)
} 


// R5 ------------------------------------------------------------------------------------
const station_R005 = IconMarker("alert","5");
function station_icon_R5(feature, latlng) {
    return L.marker(latlng, { icon: station_R005 });
}
function add_station_icon_R5(layer){
    const st = L.geoJSON(layer.features.filter(item => item.properties.alert === "R5"), {
        pointToLayer: station_icon_R5,
    });
    return(st)
} 


// R10 ------------------------------------------------------------------------------------
const station_R010 = IconMarker("alert","10");
function station_icon_R10(feature, latlng) {
    return L.marker(latlng, { icon: station_R010 });
}
function add_station_icon_R10(layer){
    const st = L.geoJSON(layer.features.filter(item => item.properties.alert === "R10"), {
        pointToLayer: station_icon_R10,
    });
    return(st)
} 