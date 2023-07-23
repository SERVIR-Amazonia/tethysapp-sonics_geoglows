// ------------------------------------------------------------------------------------------------------------ //
//                                            PANEL DATA INFORMATION                                            //
// ------------------------------------------------------------------------------------------------------------ //
var global_comid;
const sleep = ms => new Promise(r => setTimeout(r, ms));
const loader = `<div class="loading-container" style="height: 350px; padding-top: 12px;"> 
                    <div class="loading"> 
                    <h2>LOADING DATA</h2>
                    <span></span><span></span><span></span><span></span><span></span><span></span><span></span> 
                    </div>
                </div>`; 


async function showPanel(e) {
    
    // get variables of the layer
    const comid = e.layer.feature.properties.comid;
    const lat = e.layer.feature.properties.latitude;
    const lon = e.layer.feature.properties.longitude;
    const loc1 = e.layer.feature.properties.loc1;
    const loc2 = e.layer.feature.properties.loc2;

    // Show the data panel
    $("#panel-modal").modal("show")

    // Updating the comid
    global_comid = comid
    
    // Add data to the panel
    $("#panel-title-custom").html(`${comid}`)
    $("#station-comid-custom").html(`<b>COMID:</b> &nbsp ${comid}`)
    $("#station-river-custom").html(`<b>RIO:</b> &nbsp -`)
    $("#station-latitude-custom").html(`<b>LATITUD:</b> &nbsp ${lat}`)
    $("#station-longitude-custom").html(`<b>LONGITUD:</b> &nbsp ${lon}`)
    $("#station-locality1-custom").html(`<b>DEPARTAMENTO:</b> &nbsp ${loc1}`)
    $("#station-locality2-custom").html(`<b>PROVINCIA:</b> &nbsp ${loc2}`)


    // Add the dynamic loader
    $("#hydrograph").html(loader)
    $("#visual-analisis").html(loader)
    $("#metrics").html(loader)
    $("#forecast").html(loader)
    $("#corrected-forecast").html(loader)

    // We need stop 300ms to obtain the width of the panel-tab-content
    await sleep(300);

    // Retrieve the data
    $.ajax({
        type: 'GET', 
        url: "get-data",
        data: {
            comid: comid,
            fecha: end_date.replace(/-/g, ""),
            width: `${$("#panel-tab-content").width()}`
        }
    }).done(function(response){
        // Add data and plot to panel
        $("#modal-body-panel-custom").html(response)
        // Set active variables for panel data 
        active_comid = comid;
    })

}
