// ------------------------------------------------------------------------------------------------------------ //
//                                             MAP CONTROL - LAYERS                                             //
// ------------------------------------------------------------------------------------------------------------ //

// Define the control panel container
var layer_control = L.control({position:'bottomleft'}); 

// Configure the control panel container
layer_control.onAdd = function (map) {
    // Create the control panel in the DOM
    this._div = L.DomUtil.create('div', 'control')

    // Render the control panel
    this._div.innerHTML =  `
            <div class="d-flex align-items-start">
                <div class="nav flex-column nav-pills me-3" id="v-pills-tab" role="tablist" aria-orientation="vertical">
                    <button 
                        class="nav-link active" 
                        id="v-pills-layer-tab" 
                        data-bs-toggle="pill" 
                        data-bs-target="#v-pills-layer"
                        type="button" 
                        role="tab" 
                        aria-controls="v-pills-layer" 
                        aria-selected="true"> <i class="bi bi-stack"></i> </button>
                    <button 
                        class="nav-link" 
                        id="v-pills-filter-tab" 
                        data-bs-toggle="pill" 
                        data-bs-target="#v-pills-filter" 
                        type="button" 
                        role="tab" 
                        aria-controls="v-pills-filter" 
                        aria-selected="false"> <i class="bi bi-funnel-fill"></i> </button>
                    <button 
                        class="nav-link" 
                        id="v-pills-search-tab" 
                        data-bs-toggle="pill" 
                        data-bs-target="#v-pills-search" 
                        type="button" 
                        role="tab" 
                        aria-controls="v-pills-search" 
                        aria-selected="false"> <i class="bi bi-search"></i> </button>
                    <button 
                        class="nav-link" 
                        id="v-pills-upload-tab" 
                        data-bs-toggle="pill" 
                        data-bs-target="#v-pills-upload" 
                        type="button" 
                        role="tab" 
                        aria-controls="v-pills-upload" 
                        aria-selected="false"> <i class="bi bi-layer-forward"></i> </button>                    
                </div>
                <div class="tab-content" id="v-pills-tabContent">
                    <div 
                        class="tab-pane fade show active" 
                        id="v-pills-layer" 
                        role="tabpanel" 
                        aria-labelledby="v-pills-layer-tab"> ${alert_control} </div>
                    <div 
                        class="tab-pane fade" 
                        id="v-pills-filter" 
                        role="tabpanel" 
                        aria-labelledby="v-pills-filter-tab">${selbox_control}</div>
                    <div 
                        class="tab-pane fade" 
                        id="v-pills-search" 
                        role="tabpanel" 
                        aria-labelledby="v-pills-search-tab">${search_control}</div>
                    <div 
                        class="tab-pane fade" 
                        id="v-pills-upload" 
                        role="tabpanel" 
                        aria-labelledby="v-pills-upload-tab">${shp_control}</div>
                </div>
            </div>
    `;

    // Disable propagation and return
    L.DomEvent.disableClickPropagation(this._div);
    return this._div;
};

// Add the control panel container to the map
layer_control.addTo(map);

// Add dynamic functions
window.addEventListener("load", function(){
    dynamic_select_alert();
    dynamic_select_boxes();
    dynamic_search_boxes();
    insert_shapefile();
});





/* 
    <button 
        class="nav-link position-absolute bottom-0 start-0" 
        id="v-pills-close-tab" 
        data-bs-toggle="pill" 
        data-bs-target="#v-pills-close" 
        type="button" 
        role="tab" 
        aria-controls="v-pills-close" 
        aria-selected="false"> <i class="bi bi-x-lg"></i> </button>

*/