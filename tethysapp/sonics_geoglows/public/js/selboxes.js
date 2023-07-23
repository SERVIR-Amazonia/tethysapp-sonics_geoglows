// ------------------------------------------------------------------------------------------------------------ //
//                                    DATA ON LOCALITIES AND BASIN DISTRICTS                                    //
// ------------------------------------------------------------------------------------------------------------ //

// Localities
let loc = [
    { name: "Amazonas", file: "AMAZONAS.json" },
    { name: "Ancash", file: "ANCASH.json" },
    { name: "Apurimac", file: "APURIMAC.json" },
    { name: "Arequipa", file: "AREQUIPA.json" },
    { name: "Ayacucho", file: "AYACUCHO.json" },
    { name: "Cajamarca", file: "CAJAMARCA.json" },
    { name: "Callao", file: "CALLAO.json" },
    { name: "Cusco", file: "CUSCO.json" },
    { name: "Huancavelica", file: "HUANCAVELICA.json" },
    { name: "Huanuco", file: "HUANUCO.json" },
    { name: "Ica", file: "ICA.json" },
    { name: "Junin", file: "JUNIN.json" },
    { name: "La Libertad", file: "LA_LIBERTAD.json" },
    { name: "Lambayeque", file: "LAMBAYEQUE.json" },
    { name: "Lima", file: "LIMA.json" },
    { name: "Loreto", file: "LORETO.json" },
    { name: "Madre de Dios", file: "MADRE_DE_DIOS.json" },
    { name: "Moquegua", file: "MOQUEGUA.json" },
    { name: "Pasco", file: "PASCO.json" },
    { name: "Piura", file: "PIURA.json" },
    { name: "Puno", file: "PUNO.json" },
    { name: "San Martin", file: "SAN_MARTIN.json" },
    { name: "Tacna", file: "TACNA.json" },
    { name: "Tumbes", file: "TUMBES.json" },
    { name: "Ucayali", file: "UCAYALI.json" },
  ];
  
  // Localities
  let subloc = [
    { name: "0101 CHACHAPOYAS", file: "0101_CHACHAPOYAS.json" },
    { name: "0102 BAGUA", file: "0102_BAGUA.json" },
    { name: "0103 BONGARA", file: "0103_BONGARA.json" },
    { name: "0104 CONDORCANQUI", file: "0104_CONDORCANQUI.json" },
    { name: "0105 LUYA", file: "0105_LUYA.json" },
    {
      name: "0106 RODRIGUEZ DE MENDOZA",
      file: "0106_RODRIGUEZ_DE_MENDOZA.json",
    },
    { name: "0107 UTCUBAMBA", file: "0107_UTCUBAMBA.json" },
    { name: "0201 HUARAZ", file: "0201_HUARAZ.json" },
    { name: "0202 AIJA", file: "0202_AIJA.json" },
    { name: "0203 ANTONIO RAYMONDI", file: "0203_ANTONIO_RAYMONDI.json" },
    { name: "0204 ASUNCION", file: "0204_ASUNCION.json" },
    { name: "0205 BOLOGNESI", file: "0205_BOLOGNESI.json" },
    { name: "0206 CARHUAZ", file: "0206_CARHUAZ.json" },
    {
      name: "0207 CARLOS FERMIN FITZCARRALD",
      file: "0207_CARLOS_FERMIN_FITZCARRALD.json",
    },
    { name: "0208 CASMA", file: "0208_CASMA.json" },
    { name: "0209 CORONGO", file: "0209_CORONGO.json" },
    { name: "0210 HUARI", file: "0210_HUARI.json" },
    { name: "0211 HUARMEY", file: "0211_HUARMEY.json" },
    { name: "0212 HUAYLAS", file: "0212_HUAYLAS.json" },
    { name: "0213 MARISCAL LUZURIAGA", file: "0213_MARISCAL_LUZURIAGA.json" },
    { name: "0214 OCROS", file: "0214_OCROS.json" },
    { name: "0215 PALLASCA", file: "0215_PALLASCA.json" },
    { name: "0216 POMABAMBA", file: "0216_POMABAMBA.json" },
    { name: "0217 RECUAY", file: "0217_RECUAY.json" },
    { name: "0218 SANTA", file: "0218_SANTA.json" },
    { name: "0219 SIHUAS", file: "0219_SIHUAS.json" },
    { name: "0220 YUNGAY", file: "0220_YUNGAY.json" },
    { name: "0301 ABANCAY", file: "0301_ABANCAY.json" },
    { name: "0302 ANDAHUAYLAS", file: "0302_ANDAHUAYLAS.json" },
    { name: "0303 ANTABAMBA", file: "0303_ANTABAMBA.json" },
    { name: "0304 AYMARAES", file: "0304_AYMARAES.json" },
    { name: "0305 COTABAMBAS", file: "0305_COTABAMBAS.json" },
    { name: "0306 CHINCHEROS", file: "0306_CHINCHEROS.json" },
    { name: "0307 GRAU", file: "0307_GRAU.json" },
    { name: "0401 AREQUIPA", file: "0401_AREQUIPA.json" },
    { name: "0402 CAMANA", file: "0402_CAMANA.json" },
    { name: "0403 CARAVELI", file: "0403_CARAVELI.json" },
    { name: "0404 CASTILLA", file: "0404_CASTILLA.json" },
    { name: "0405 CAYLLOMA", file: "0405_CAYLLOMA.json" },
    { name: "0406 CONDESUYOS", file: "0406_CONDESUYOS.json" },
    { name: "0407 ISLAY", file: "0407_ISLAY.json" },
    { name: "0408 LA UNION", file: "0408_LA_UNION.json" },
    { name: "0501_HUAMANGA", file: "0501_HUAMANGA.json" },
    { name: "0502 CANGALLO", file: "0502_CANGALLO.json" },
    { name: "0503 HUANCA SANCOS", file: "0503_HUANCA_SANCOS.json" },
    { name: "0504 HUANTA", file: "0504_HUANTA.json" },
    { name: "0505 LA MAR", file: "0505_LA_MAR.json" },
    { name: "0506 LUCANAS", file: "0506_LUCANAS.json" },
    { name: "0507 PARINACOCHAS", file: "0507_PARINACOCHAS.json" },
    {
      name: "0508 PAUCAR DEL SARA SARA",
      file: "0508_PAUCAR_DEL_SARA_SARA.json",
    },
    { name: "0509 SUCRE", file: "0509_SUCRE.json" },
    { name: "0510 VICTOR FAFARDO", file: "0510_VICTOR_FAFARDO.json" },
    { name: "0510 VICTOR FAJARDO", file: "0510_VICTOR_FAJARDO.json" },
    { name: "0511 VILCAS HUAMAN", file: "0511_VILCAS_HUAMAN.json" },
    { name: "0601 CAJAMARCA", file: "0601_CAJAMARCA.json" },
    { name: "0602 CAJABAMBA", file: "0602_CAJABAMBA.json" },
    { name: "0603 CELENDIN", file: "0603_CELENDIN.json" },
    { name: "0604 CHOTA", file: "0604_CHOTA.json" },
    { name: "0605 CONTUMAZA", file: "0605_CONTUMAZA.json" },
    { name: "0606 CUTERVO", file: "0606_CUTERVO.json" },
    { name: "0607 HUALGAYOC", file: "0607_HUALGAYOC.json" },
    { name: "0608 JAEN", file: "0608_JAEN.json" },
    { name: "0609 SAN IGNACIO", file: "0609_SAN_IGNACIO.json" },
    { name: "0610 SAN MARCOS", file: "0610_SAN_MARCOS.json" },
    { name: "0611 SAN MIGUEL", file: "0611_SAN_MIGUEL.json" },
    { name: "0612 SAN PABLO", file: "0612_SAN_PABLO.json" },
    { name: "0613 SANTA CRUZ", file: "0613_SANTA_CRUZ.json" },
    { name: "0701 CALLAO", file: "0701_CALLAO.json" },
    { name: "0801 CUSCO", file: "0801_CUSCO.json" },
    { name: "0802 ACOMAYO", file: "0802_ACOMAYO.json" },
    { name: "0803 ANTA", file: "0803_ANTA.json" },
    { name: "0804 CALCA", file: "0804_CALCA.json" },
    { name: "0805 CANAS", file: "0805_CANAS.json" },
    { name: "0806 CANCHIS", file: "0806_CANCHIS.json" },
    { name: "0807 CHUMBIVILCAS", file: "0807_CHUMBIVILCAS.json" },
    { name: "0808 ESPINAR", file: "0808_ESPINAR.json" },
    { name: "0809 LA CONVENCION", file: "0809_LA_CONVENCION.json" },
    { name: "0810 PARURO", file: "0810_PARURO.json" },
    { name: "0811 PAUCARTAMBO", file: "0811_PAUCARTAMBO.json" },
    { name: "0812 QUISPICANCHI", file: "0812_QUISPICANCHI.json" },
    { name: "0813 URUBAMBA", file: "0813_URUBAMBA.json" },
    { name: "0901 HUANCAVELICA", file: "0901_HUANCAVELICA.json" },
    { name: "0902 ACOBAMBA", file: "0902_ACOBAMBA.json" },
    { name: "0903 ANGARAES", file: "0903_ANGARAES.json" },
    { name: "0904 CASTROVIRREYNA", file: "0904_CASTROVIRREYNA.json" },
    { name: "0905 CHURCAMPA", file: "0905_CHURCAMPA.json" },
    { name: "0906 HUAYTARA", file: "0906_HUAYTARA.json" },
    { name: "0907 TAYACAJA", file: "0907_TAYACAJA.json" },
    { name: "1001 HUANUCO", file: "1001_HUANUCO.json" },
    { name: "1002 AMBO", file: "1002_AMBO.json" },
    { name: "1003 DOS DE MAYO", file: "1003_DOS_DE_MAYO.json" },
    { name: "1004 HUACAYBAMBA", file: "1004_HUACAYBAMBA.json" },
    { name: "1005 HUAMALIES", file: "1005_HUAMALIES.json" },
    { name: "1006 LEONCIO PRADO", file: "1006_LEONCIO_PRADO.json" },
    { name: "1007 MARANON", file: "1007_MARANON.json" },
    { name: "1008 PACHITEA", file: "1008_PACHITEA.json" },
    { name: "1009 PUERTO INCA", file: "1009_PUERTO_INCA.json" },
    { name: "1010 LAURICOCHA", file: "1010_LAURICOCHA.json" },
    { name: "1011 YAROWILCA", file: "1011_YAROWILCA.json" },
    { name: "1101 ICA", file: "1101_ICA.json" },
    { name: "1102 CHINCHA", file: "1102_CHINCHA.json" },
    { name: "1103 NAZCA", file: "1103_NAZCA.json" },
    { name: "1104 PALPA", file: "1104_PALPA.json" },
    { name: "1105 PISCO", file: "1105_PISCO.json" },
    { name: "1201 HUANCAYO", file: "1201_HUANCAYO.json" },
    { name: "1202 CONCEPCION", file: "1202_CONCEPCION.json" },
    { name: "1203 CHANCHAMAYO", file: "1203_CHANCHAMAYO.json" },
    { name: "1204 JAUJA", file: "1204_JAUJA.json" },
    { name: "1205 JUNIN", file: "1205_JUNIN.json" },
    { name: "1206 SATIPO", file: "1206_SATIPO.json" },
    { name: "1207 TARMA", file: "1207_TARMA.json" },
    { name: "1208 YAULI", file: "1208_YAULI.json" },
    { name: "1209 CHUPACA", file: "1209_CHUPACA.json" },
    { name: "1301 TRUJILLO", file: "1301_TRUJILLO.json" },
    { name: "1302 ASCOPE", file: "1302_ASCOPE.json" },
    { name: "1303 BOLIVAR", file: "1303_BOLIVAR.json" },
    { name: "1304 CHEPEN", file: "1304_CHEPEN.json" },
    { name: "1305 JULCAN", file: "1305_JULCAN.json" },
    { name: "1306 OTUZCO", file: "1306_OTUZCO.json" },
    { name: "1307 PACASMAYO", file: "1307_PACASMAYO.json" },
    { name: "1308 PATAZ", file: "1308_PATAZ.json" },
    { name: "1309 SANCHEZ CARRION", file: "1309_SANCHEZ_CARRION.json" },
    { name: "1310 SANTIAGO DE CHUCO", file: "1310_SANTIAGO_DE_CHUCO.json" },
    { name: "1311 GRAN CHIMU", file: "1311_GRAN_CHIMU.json" },
    { name: "1312 VIRU", file: "1312_VIRU.json" },
    { name: "1401 CHICLAYO", file: "1401_CHICLAYO.json" },
    { name: "1402 FERRENAFE", file: "1402_FERRENAFE.json" },
    { name: "1403 LAMBAYEQUE", file: "1403_LAMBAYEQUE.json" },
    { name: "1501 LIMA", file: "1501_LIMA.json" },
    { name: "1502 BARRANCA", file: "1502_BARRANCA.json" },
    { name: "1503 CAJATAMBO", file: "1503_CAJATAMBO.json" },
    { name: "1504 CANTA", file: "1504_CANTA.json" },
    { name: "1505 CANETE", file: "1505_CANETE.json" },
    { name: "1506 HUARAL", file: "1506_HUARAL.json" },
    { name: "1507 HUAROCHIRI", file: "1507_HUAROCHIRI.json" },
    { name: "1508 HUAURA", file: "1508_HUAURA.json" },
    { name: "1509 OYON", file: "1509_OYON.json" },
    { name: "1510 YAUYOS", file: "1510_YAUYOS.json" },
    { name: "1601 MAYNAS", file: "1601_MAYNAS.json" },
    { name: "1602 ALTO AMAZONAS", file: "1602_ALTO_AMAZONAS.json" },
    { name: "1603 LORETO", file: "1603_LORETO.json" },
    {
      name: "1604 MARISCAL RAMON CASTILLA",
      file: "1604_MARISCAL_RAMON_CASTILLA.json",
    },
    { name: "1605 REQUENA", file: "1605_REQUENA.json" },
    { name: "1606 UCAYALI", file: "1606_UCAYALI.json" },
    { name: "1607 DATEM DEL MARANON", file: "1607_DATEM_DEL_MARANON.json" },
    { name: "1701 TAMBOPATA", file: "1701_TAMBOPATA.json" },
    { name: "1702 MANU", file: "1702_MANU.json" },
    { name: "1703 TAHUAMANU", file: "1703_TAHUAMANU.json" },
    { name: "1801 MARISCAL NIETO", file: "1801_MARISCAL_NIETO.json" },
    {
      name: "1802 GENERAL SANCHEZ CERRO",
      file: "1802_GENERAL_SANCHEZ_CERRO.json",
    },
    { name: "1803 ILO", file: "1803_ILO.json" },
    { name: "1901 PASCO", file: "1901_PASCO.json" },
    {
      name: "1902 DANIEL ALCIDES CARRION",
      file: "1902_DANIEL_ALCIDES_CARRION.json",
    },
    { name: "1903 OXAPAMPA", file: "1903_OXAPAMPA.json" },
    { name: "2001 PIURA", file: "2001_PIURA.json" },
    { name: "2001 PUIRA", file: "2001_PUIRA.json" },
    { name: "2002 AYABACA", file: "2002_AYABACA.json" },
    { name: "2003 HUANCABAMBA", file: "2003_HUANCABAMBA.json" },
    { name: "2004 MORROPON", file: "2004_MORROPON.json" },
    { name: "2005 PAITA", file: "2005_PAITA.json" },
    { name: "2006 SULLANA", file: "2006_SULLANA.json" },
    { name: "2007 TALARA", file: "2007_TALARA.json" },
    { name: "2008 SECHURA", file: "2008_SECHURA.json" },
    { name: "2101 PUNO", file: "2101_PUNO.json" },
    { name: "2102 AZANGARO", file: "2102_AZANGARO.json" },
    { name: "2103 CARABAYA", file: "2103_CARABAYA.json" },
    { name: "2104 CHUCUITO", file: "2104_CHUCUITO.json" },
    { name: "2105 EL COLLAO", file: "2105_EL_COLLAO.json" },
    { name: "2106 HUANCANE", file: "2106_HUANCANE.json" },
    { name: "2107 LAMPA", file: "2107_LAMPA.json" },
    { name: "2108 MELGAR", file: "2108_MELGAR.json" },
    { name: "2109 MOHO", file: "2109_MOHO.json" },
    {
      name: "2110 SAN ANTONIO DE PUTINA",
      file: "2110_SAN_ANTONIO_DE_PUTINA.json",
    },
    { name: "2111 SAN ROMAN", file: "2111_SAN_ROMAN.json" },
    { name: "2112 SANDIA", file: "2112_SANDIA.json" },
    { name: "2113 YUNGUYO", file: "2113_YUNGUYO.json" },
    { name: "2201 MOYOBAMBA", file: "2201_MOYOBAMBA.json" },
    { name: "2202 BELLAVISTA", file: "2202_BELLAVISTA.json" },
    { name: "2203 EL DORADO", file: "2203_EL_DORADO.json" },
    { name: "2204 HUALLAGA", file: "2204_HUALLAGA.json" },
    { name: "2205 LAMAS", file: "2205_LAMAS.json" },
    { name: "2206 MARISCAL CACERES", file: "2206_MARISCAL_CACERES.json" },
    { name: "2207 PICOTA", file: "2207_PICOTA.json" },
    { name: "2208 RIOJA", file: "2208_RIOJA.json" },
    { name: "2209 SAN MARTIN", file: "2209_SAN_MARTIN.json" },
    { name: "2210 TOCACHE", file: "2210_TOCACHE.json" },
    { name: "2301 TACNA", file: "2301_TACNA.json" },
    { name: "2302 CANDARAVE", file: "2302_CANDARAVE.json" },
    { name: "2303 JORGE BASADRE", file: "2303_JORGE_BASADRE.json" },
    { name: "2304 TARATA", file: "2304_TARATA.json" },
    { name: "2401 TUMBES", file: "2401_TUMBES.json" },
    {
      name: "2402 CONTRALMIRANTE VILLAR",
      file: "2402_CONTRALMIRANTE_VILLAR.json",
    },
    { name: "2403 ZARUMILLA", file: "2403_ZARUMILLA.json" },
    { name: "2501 CORONEL PORTILLO", file: "2501_CORONEL_PORTILLO.json" },
    { name: "2502 ATALAYA", file: "2502_ATALAYA.json" },
    { name: "2503 PADRE ABAD", file: "2503_PADRE_ABAD.json" },
    { name: "2504 PURUS", file: "2504_PURUS.json" },
  ];
  
  // River basin districts
  let basin = [
    { name: "0144 Mauri", file: "0144_Mauri.json" },
    { name: "0146 Cano", file: "0146_Cano.json" },
    { name: "0148 Uchusuma", file: "0148_Uchusuma.json" },
    { name: "0152 Mauri Chico", file: "0152_Mauri_Chico.json" },
    {
      name: "0155 Unidad hidrografica 0155",
      file: "0155_Unidad_hidrografica_0155.json",
    },
    { name: "0156 Callaccame", file: "0156_Callaccame.json" },
    {
      name: "0157 Unidad hidrografica 0157",
      file: "0157_Unidad_hidrografica_0157.json",
    },
    { name: "015 Lago Titicaca", file: "015_Lago_Titicaca.json" },
    { name: "016 Cuenca Ilave", file: "016_Cuenca_Ilave.json" },
    {
      name: "0171 Unidad Hidrografica 0171",
      file: "0171_Unidad_Hidrografica_0171.json",
    },
    { name: "0172 Suches", file: "0172_Suches.json" },
    {
      name: "0173 Unidad Hidrografica 0173",
      file: "0173_Unidad_Hidrografica_0173.json",
    },
    { name: "0174 Ilpa", file: "0174_Ilpa.json" },
    {
      name: "0175 Unidad Hidrografica 0175",
      file: "0175_Unidad_Hidrografica_0175.json",
    },
    { name: "0176 Coata", file: "0176_Coata.json" },
    { name: "0178 Huancane", file: "0178_Huancane.json" },
    {
      name: "0179 Unidad Hidrografica 0179",
      file: "0179_Unidad_Hidrografica_0179.json",
    },
    { name: "017 Lago Titicaca", file: "017_Lago_Titicaca.json" },
    { name: "018 Cuenca Pucara", file: "018_Cuenca_Pucara.json" },
    { name: "019 Cuenca Azangaro", file: "019_Cuenca_Azangaro.json" },
    { name: "1314 Lluta", file: "1314_Lluta.json" },
    {
      name: "1315_Unidad_Hidrografica_1315",
      file: "1315 Unidad Hidrografica 1315.json",
    },
    { name: "1316 Locumba", file: "1316_Locumba.json" },
    {
      name: "1317 Unidad Hidrografica 1317",
      file: "1317_Unidad_Hidrografica_1317.json",
    },
    { name: "1318 Tambo", file: "1318_Tambo.json" },
    {
      name: "1319 Unidad Hidrografica 1319",
      file: "1319_Unidad_Hidrografica_1319.json",
    },
    {
      name: "132 Cuenca Quilca Vitor Chili",
      file: "132_Cuenca_Quilca_Vitor_Chili.json",
    },
    { name: "133 Intercuenca 133", file: "133_Intercuenca_133.json" },
    { name: "134 Cuenca Camana", file: "134_Cuenca_Camana.json" },
    { name: "135 Intercuenca 135", file: "135_Intercuenca_135.json" },
    { name: "136 Cuenca Ocona", file: "136_Cuenca_Ocona.json" },
    {
      name: "1371 Unidad Hidrografica 1371",
      file: "1371_Unidad_Hidrografica_1371.json",
    },
    { name: "1372 Grande", file: "1372_Grande.json" },
    {
      name: "1373 Unidad Hidrografica 1373",
      file: "1373_Unidad_Hidrografica_1373.json",
    },
    { name: "1374 Ica", file: "1374_Ica.json" },
    {
      name: "1375 Unidad Hidrografica 1375",
      file: "1375_Unidad_Hidrografica_1375.json",
    },
    { name: "1376 Santa", file: "1376_Santa.json" },
    {
      name: "1377 Unidad Hidrografica 1377",
      file: "1377_Unidad_Hidrografica_1377.json",
    },
    { name: "1378 Piura", file: "1378_Piura.json" },
    {
      name: "1379 Unidad Hidrografica 1379",
      file: "1379_Unidad_Hidrografica_1379.json",
    },
    { name: "138 Cuenca Chira", file: "138_Cuenca_Chira.json" },
    {
      name: "1391 Unidad Hidrografica 1391",
      file: "1391_Unidad_Hidrografica_1391.json",
    },
    { name: "1392 Parinas", file: "1392_Parinas.json" },
    {
      name: "1393 Unidad Hidrografica 1393",
      file: "1393_Unidad_Hidrografica_1393.json",
    },
    { name: "1394 Tumbes", file: "1394_Tumbes.json" },
    {
      name: "1395 Unidad Hidrografica 1395",
      file: "1395_Unidad_Hidrografica_1395.json",
    },
    { name: "4662 Orthon", file: "4662_Orthon.json" },
    { name: "4664 Madre de Dios", file: "4664_Madre_de_Dios.json" },
    { name: "4926 Acre", file: "4926_Acre.json" },
    { name: "4928 Iaco", file: "4928_Iaco.json" },
    { name: "4929 Alto Purus", file: "4929_Alto_Purus.json" },
    { name: "4964 Tarau", file: "4964_Tarau.json" },
    { name: "4969 Alto Yurua", file: "4969_Alto_Yurua.json" },
    { name: "4974 Putumayo", file: "4974_Putumayo.json" },
    { name: "4976 Yavari", file: "4976_Yavari.json" },
    {
      name: "4977 Unidad_Hidrografica 4977",
      file: "4977_Unidad_Hidrografica_4977.json",
    },
    { name: "4978 Napo", file: "4978_Napo.json" },
    {
      name: "4979 Unidad Hidrografica 4979",
      file: "4979_Unidad_Hidrografica_4979.json",
    },
    { name: "4981 Bajo Maranon", file: "4981_Bajo_Maranon.json" },
    { name: "4982 Tigre", file: "4982_Tigre.json" },
    { name: "4983 Medio Bajo Maranon", file: "4983_Medio_Bajo_Maranon.json" },
    { name: "4984 Huallaga", file: "4984_Huallaga.json" },
    { name: "4985 Medio Maranon", file: "4985_Medio_Maranon.json" },
    { name: "4986 Pastaza", file: "4986_Pastaza.json" },
    { name: "4987 Medio Alto Maranon", file: "4987_Medio_Alto_Maranon.json" },
    { name: "4988 Cenepa", file: "4988_Cenepa.json" },
    { name: "4989 Alto Maranon", file: "4989_Alto_Maranon.json" },
    { name: "4991 Bajo Ucayali", file: "4991_Bajo_Ucayali.json" },
    { name: "4992 Pachitea", file: "4992_Pachitea.json" },
    { name: "4993 Medio Bajo Ucayali", file: "4993_Medio_Bajo_Ucayali.json" },
    { name: "4994 Urubamba", file: "4994_Urubamba.json" },
    { name: "4995 Medio Ucayali", file: "4995_Medio_Ucayali.json" },
    { name: "4996 Mantaro", file: "4996_Mantaro.json" },
    { name: "4997 Medio Alto Ucayali", file: "4997_Medio_Alto_Ucayali.json" },
    { name: "4998 Pampas", file: "4998_Pampas.json" },
    { name: "4999 Alto Ucayali", file: "4999_Alto_Ucayali.json" },
  ];

  
// ------------------------------------------------------------------------------------------------------------ //
//                                    DATA ON LOCALITIES AND BASIN DISTRICTS                                    //
// ------------------------------------------------------------------------------------------------------------ //

// Generate options for Localities
const loc_url = `${server}/static/${app_name}/geojson/loc/`
loc = loc.map((item) => {
    var option_custom = `<option value="${item.file}">${item.name}</option>`;
    return(option_custom);
}).join("");


// Generate options for Sublocalities
const subloc_url = `${server}/static/${app_name}/geojson/subloc/`
subloc = subloc.map((item) => {
    var option_custom = `<option value="${item.file}">${item.name}</option>`;
    return(option_custom);
}).join("");


// Generate options for River basin districts
const basin_url = `${server}/static/${app_name}/geojson/basin/`
basin = basin.map((item) => {
    var option_custom = `<option value="${item.file}">${item.name}</option>`;
    return(option_custom);
}).join("");



selbox_control = `
    <div class="control-group">
        <label class="label-control" for="select-loc">Departamento:</label>
        <select id="select-loc" required class="demo-default" placeholder="Seleccione un departamento." name="loc">
            <option value="">Seleccione un departamento.</option>
            ${loc}
        </select>
        <br>
        <label class="label-control" for="select-subloc">Provincia:</label>
        <select id="select-subloc" required class="demo-default" placeholder="Seleccione una provincia." name="subloc">
            <option value="">Seleccione una provincia.</option>
            ${subloc}
        </select>
        <br>
        <label class="label-control" for="select-basin">Cuenca Hidrográfica:</label>
        <select id="select-basin" required class="demo-default" placeholder="Seleccione una demarcación hidrográfica." name="basin">
            <option value="">Seleccione una demarcación hidrográfica.</option>
            ${basin}
        </select>
        <br>
    </div>`;


function dynamic_select_boxes(){
    // Select box for ZOOM to localities (Departamentos)
    $('#select-loc').selectize({
        create: false,
        //sortField: { field: 'text', direction: 'asc'},
        onChange: function(value, isOnInitialize) {
            // Retrieve geojson from REST API
            fetch(`${loc_url}${value}`)
            .then((response) => (layer = response.json()))
            .then((layer) => {
                // Remove the current layer
                if (typeof layerSHP !== 'undefined') {
                    map.removeLayer(layerSHP)
                }
                // Add retrieved layer and fit to map
                if(value === "peru.geojson"){
                    layerSHP = L.geoJSON(layer, { style:  {weight: 2, fillOpacity: 0} }).addTo(map);
                }else{
                    layerSHP = L.geoJSON(layer, { style: { weight: 1 } }).addTo(map);
                }
                map.fitBounds(layerSHP.getBounds());
            });
        }
    });


    // Select box for ZOOM to localities (Provincias)
    $('#select-subloc').selectize({
        create: false,
        //sortField: { field: 'text', direction: 'asc'},
        onChange: function(value, isOnInitialize) {
            // Retrieve geojson from REST API
            fetch(`${subloc_url}${value}`)
            .then((response) => (layer = response.json()))
            .then((layer) => {
                // Remove the current layer
                if (typeof layerSHP !== 'undefined') {
                    map.removeLayer(layerSHP)
                }
                // Add retrieved layer and fit to map
                if(value === "peru.geojson"){
                    layerSHP = L.geoJSON(layer, { style:  {weight: 2, fillOpacity: 0} }).addTo(map);
                }else{
                    layerSHP = L.geoJSON(layer, { style: { weight: 1 } }).addTo(map);
                }
                map.fitBounds(layerSHP.getBounds());
            });
        }
    });



    // Select box for ZOOM to to basin district
    $('#select-basin').selectize({
        create: true,
        //sortField: { field: 'text', direction: 'asc'},
        onChange: function(value, isOnInitialize) {
            // Retrieve geojson from REST API
            fetch(`${basin_url}${value}`)
            .then((response) => (layer = response.json()))
            .then((layer) => {
                // Remove the current layer
                if (typeof layerSHP !== 'undefined') {
                    map.removeLayer(layerSHP)
                }
                // Add retrieved layer and fit to map
                if(value === "peru.geojson"){
                    layerSHP = L.geoJSON(layer, { style:  {weight: 2, fillOpacity: 0} }).addTo(map);
                }else{
                    layerSHP = L.geoJSON(layer, { style: { weight: 1 } }).addTo(map);
                }
                map.fitBounds(layerSHP.getBounds());
            });
        }
    });
}