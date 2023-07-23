alert_control =  `
        <div class="control-group"> 
            <label class="label-control" for="select-loc">Niveles de alerta:</label>
            <div class="alert-panel-checkbox">
                <div class="form-check form-switch">
                    <input class="form-check-input" type="checkbox" id="check-002yr" checked>
                    <label class="form-check-label" for="check-002yr">Periodo de retorno: 2.33 años</label>
                </div>
                <div class="form-check form-switch">
                    <input class="form-check-input" type="checkbox" id="check-005yr" checked>
                    <label class="form-check-label" for="check-005yr">Periodo de retorno: 5 años</label>
                </div>
                <div class="form-check form-switch">
                    <input class="form-check-input" type="checkbox" id="check-010yr" checked>
                    <label class="form-check-label" for="check-010yr">Periodo de retorno: 10 años</label>
                </div>
            </div>
        </div>`;



function dynamic_select_alert(){

    $('#check-000yr').on('change', function () {
        if($('#check-000yr').is(':checked')){
            est_R000.addTo(map);
        } else {
            map.removeLayer(est_R000); 
        };
    });
            
    $('#check-002yr').on('change', function () {
        if($('#check-002yr').is(':checked')){
            est_R002.addTo(map);
        } else {
            map.removeLayer(est_R002); 
        };
    });
            
    $('#check-005yr').on('change', function () {
        if($('#check-005yr').is(':checked')){
            est_R005.addTo(map);
        } else {
            map.removeLayer(est_R005); 
        };
    });
            
    $('#check-010yr').on('change', function () {
        if($('#check-010yr').is(':checked')){
            est_R010.addTo(map);
        } else {
            map.removeLayer(est_R010); 
        };
    });
            
    $('#check-025yr').on('change', function () {
        if($('#check-025yr').is(':checked')){
            est_R025.addTo(map);
        } else {
            map.removeLayer(est_R025); 
        };
    });
            
    $('#check-050yr').on('change', function () {
        if($('#check-050yr').is(':checked')){
            est_R050.addTo(map);
        } else {
            map.removeLayer(est_R050); 
        };
    });
            
    $('#check-100yr').on('change', function () {
        if($('#check-100yr').is(':checked')){
            est_R100.addTo(map);
        } else {
            map.removeLayer(est_R100); 
        };
    });
}
