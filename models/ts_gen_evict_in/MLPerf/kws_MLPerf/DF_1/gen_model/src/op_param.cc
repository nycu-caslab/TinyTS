#include "gen_model/include/op_param.h"

const SharedParam_AvgPool shared_param_avgpool[1] = {
    {35, 36, 25, 5, 25, 5, 1, 0}
};
const SharedParam_Concat shared_param_concat[1] = {
    {34, 35, 1}
};
const SharedParam_Conv shared_param_conv[5] = {
    {16, 1, 6, 17, 0, 1, 0, 2, 2, 1, 1},
    {21, 2, 7, 22, 0, 1, 352, 1, 1, 1, 1},
    {25, 3, 8, 26, 0, 1, 768, 1, 1, 1, 1},
    {29, 4, 9, 30, 0, 1, 1184, 1, 1, 1, 1},
    {33, 5, 10, 34, 0, 1, 1600, 1, 1, 1, 1}
};
const SharedParam_Depthwise_Conv shared_param_dwconv[4] = {
    {17, 18, 19, 21, 1, 0, 1, 144, 1, 1, 1, 1},
    {22, 23, 24, 25, 1, 0, 1, 560, 1, 1, 1, 1},
    {26, 27, 28, 29, 1, 0, 1, 976, 1, 1, 1, 1},
    {30, 31, 32, 33, 1, 0, 1, 1392, 1, 1, 1, 1}
};
const SharedParam_FC shared_param_fc[1] = {
    {38, 39, 40, 41, 0, 0, false, false}
};
const SharedParam_Reshape shared_param_reshape[1] = {
    {36, 38}
};
const SharedParam_Softmax shared_param_softmax[1] = {
    {41, 42, 1065353216}
};
const SharedParam_Split shared_param_split[1] = {
    {0, 16, 49, 1}
};

alignas(16) const int32_t op_data[1808] = {
    4, 1, -128, 127, 1359514674, 1510897544, 1926649374, 1084580191, 1833203381, 1598446167, 1627679308, 1382312008, 1075542397, 1103690228, 1943762883, 1106996425, 1152722049, 1245202516, 1402695826, 1573799777, 1613521674, 1298743846, 1621288420, 1098453574, 1836254561, 1961975843, 1733989666, 1923153043, 1657760567, 1486396369, 1935422644, 1649163646, 2062717049, 1103484170, 1211514054, 1955102371, 1415290081, 1113330866, 1577332471, 1377337383, 1312594413, 1821197710, 1295475795, 1674991602, 1563913152, 1204217771, 1095865610, 2064778105, 1211410907, 1243172833, 1358168642, 2147314432, 1520073423, 1591120175, 1505197082, 2051158788, 2068548467, 1677282482, 1584536658, 1837833626, 1684233315, 1285042891, 1639420454, 1724598460, 1253095821, 2006205585, 1857196660, 1683081220, -6, -7, -8, -7, -8, -7, -6, -7, -7, -8, -7, -7, -6, -7, -7, -8, -6, -8, -8, -7, -8, -7, -8, -6, -8, -7, -8, -7, -7, -6, -7, -7, -7, -7, -7, -7, -6, -8, -8, -8, -8, -7, -7, -6, -6, -6, -7, -8, -7, -6, -7, -7, -7, -8, -7, -6, -8, -7, -7, -8, -7, -8, -7, -8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, -128, 127, 1117272080, 1351620567, 1208841889, 1680313023, 1480831180, 1867036575, 1274068427, 1551573551, 1294969724, 1383317958, 1236810790, 1458694841, 1431041453, 1527668729, 1972269723, 1085649278, 1145655177, 1883008410, 1699922209, 1787116075, 2034358525, 1870523896, 1303360346, 1162227620, 1943854503, 1401822542, 1203348201, 1837357352, 1599269182, 1757899597, 1081471549, 1540510979, 1264653146, 1386686886, 1200409896, 1368802621, 1087238284, 1193358838, 1651991982, 1445986925, 1245546981, 1617319173, 2142376177, 1394652726, 2095469635, 1151724771, 2022842212, 1433810868, 1078765408, 1334169966, 1343470626, 1463716779, 1467283313, 1962296422, 1729111065, 1720207323, 1596415686, 1989330707, 1124519040, 1702538309, 1312306067, 1708501555, 1537057119, 1801960675, -6, -7, -6, -6, -6, -6, -6, -7, -6, -7, -6, -6, -6, -6, -7, -6, -6, -6, -6, -6, -6, -7, -5, -6, -6, -7, -6, -6, -6, -6, -6, -7, -6, -6, -6, -6, -6, -7, -6, -6, -5, -6, -6, -6, -7, -6, -7, -7, -5, -6, -6, -5, -6, -7, -7, -7, -6, -7, -5, -6, -6, -7, -6, -6, -18048, -54912, -50688, 29696, 11904, 6016, -19968, -50816, 2944, 6400, -9856, 1536, -11776, -22912, -640, -384, 20608, -43904, 27008, 5248, -17920, -33792, 6144, 128, -38784, -17664, -41984, 5888, 19456, 11264, 8192, -41088, -1024, 16768, -5632, -38912, -15232, -20352, -39424, 1792, -11008, 8320, 2432, -22912, -14720, -16128, -47360, -45312, 13696, -9472, -30080, 7936, -16896, -18304, 2304, -4224, 10368, -17024, 6912, -42880, -32640, -15360, -20736, 14464, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -128, 127, 1753010496, 1400798191, 1134457379, 1758672353, 1403607575, 1820785010, 1513526197, 1246474055, 1155777242, 1479158775, 1270449839, 1121960977, 1578196323, 1171900227, 1238656486, 2129490323, 1859006295, 1150855943, 1406136170, 1096281921, 1179667643, 1135417887, 1793992513, 1262616200, 2106404761, 1285768604, 1828269871, 1969992531, 2095390798, 1276555571, 1406488921, 1388277772, 1109336278, 2041424002, 1524387405, 1247844619, 1254068671, 1466156336, 1436942880, 1083956729, 1563114285, 1409961159, 1165600769, 1500316611, 1312999944, 1194430041, 1485904580, 1348061566, 1742367209, 1178477383, 1161423647, 2135868985, 2020513224, 1409196056, 1424482857, 1335577791, 1530524306, 1106321337, 1074189536, 1740004173, 1291847582, 1929897381, 1205240477, 1231837482, -8, -7, -7, -8, -7, -8, -8, -7, -7, -7, -7, -7, -7, -7, -7, -8, -8, -7, -7, -7, -7, -7, -7, -7, -8, -7, -8, -8, -8, -7, -7, -7, -7, -8, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7, -8, -7, -8, -7, -7, -8, -8, -7, -7, -7, -8, -7, -7, -8, -7, -8, -7, -7, -63360, 75392, -23296, -52480, -76416, -135808, 160640, -69632, 6400, -22144, -42624, -47360, -54656, -1792, -96768, -33664, -14080, -25728, -124416, -2048, -43776, -119424, -54016, 86784, -83200, -26624, -93696, 81152, -11776, 11520, -10880, 21376, -94592, 29184, -29568, -103552, -116480, -4736, -39808, 121088, -39168, 27008, -64384, 35712, -37504, -39040, -62336, 51072, -16384, -24064, -62336, -136960, -36736, 16128, -9216, 11136, -76928, -66176, -43008, -22272, 23680, 71552, 15488, -14976, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, -128, 127, 1714133752, 1698805621, 1693693347, 1216969937, 1735814305, 1879620052, 1554176332, 1477316785, 1624124963, 1738267546, 1951688197, 1963805900, 1856832425, 1796474680, 1162897160, 1139020700, 2040819946, 2076564242, 1735670948, 1418463255, 1231094892, 1214681724, 1867559032, 1754692957, 2044652774, 1231630617, 2044833104, 1525740462, 1851553165, 1106654797, 1428586437, 1724402251, 1087753987, 1823558753, 1714680984, 1468675047, 1284410604, 1423009689, 2053024691, 1324499214, 1147243383, 1361934788, 2122974424, 1098402732, 1640228278, 1203990620, 1420131027, 1685960602, 1863302374, 1544851228, 1476837620, 1731253058, 1080317016, 1798928655, 1508177126, 1882108061, 1282398210, 1240758620, 1290956455, 1958000465, 1951353125, 1928126488, 1748347635, 1966609638, -7, -7, -6, -6, -7, -6, -7, -7, -6, -7, -7, -7, -7, -7, -6, -6, -7, -7, -7, -7, -7, -5, -7, -7, -7, -6, -7, -7, -7, -6, -7, -7, -5, -7, -7, -6, -7, -7, -7, -7, -6, -6, -7, -7, -7, -6, -6, -7, -7, -7, -6, -6, -6, -7, -7, -7, -6, -6, -6, -7, -7, -7, -7, -7, 56704, -50560, -1024, 39168, 10880, -36992, 20096, 7168, 22016, -36480, 61952, 14464, -3968, 75520, 34304, -4608, 23680, 77440, 40320, -70272, 22400, -37632, 26240, -32000, 39424, 31232, -7680, 57728, 41216, 35840, -30592, -2432, 37504, -8576, -31744, 20352, -65152, 0, 40960, 28288, 17024, 1408, -57984, 11264, -78592, 23168, 8448, 24192, 1792, 25984, -27264, 4608, -60160, -65920, -52224, 10624, 52480, -10752, 1792, -66048, 39424, -41728, 9088, -51072, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -128, 127, 1352513846, 1489513224, 1379458316, 1477987624, 1639906495, 1084564680, 1893259625, 1414921633, 1590248680, 1261327875, 2121746637, 1291550477, 1193854958, 1700510853, 1354542153, 1237655415, 1859562043, 2134661778, 1759378903, 2015455487, 2043579655, 1092662229, 1222915475, 1160175847, 1712718927, 1552760599, 1404169791, 1451816696, 1314090277, 1296517021, 1331039802, 1615117376, 1911710593, 1654589195, 1629185796, 1805117887, 1298266647, 1400135407, 1504218799, 1616550840, 1442929496, 1696869415, 1680909935, 1306538278, 1796114167, 1514469122, 1469406920, 1610711739, 1131109424, 1317781222, 1385498455, 1427648409, 1611025645, 1803660582, 1284534792, 1560836992, 1385906544, 1243604164, 1528928555, 1754894440, 1454913766, 1612589482, 1349712104, 1345755257, -7, -7, -7, -7, -7, -6, -7, -7, -7, -7, -7, -7, -7, -7, -6, -7, -7, -8, -7, -8, -7, -7, -7, -6, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7, -72832, 114048, 43264, 95616, -40960, -77312, 95616, -32640, -13952, -42240, -42496, -165632, -75904, 19072, -76800, -77696, 25216, -61696, -92416, 16000, 41728, -22528, 66816, 1664, 116480, 26112, 88576, -136960, -9344, -46208, 20096, 20864, -54144, -9344, -11264, -30080, -72320, 12544, -64512, 32896, 19840, -27904, -1920, 96768, -5888, 16128, 61824, -37632, -37376, -13440, 17280, -58496, 43264, -72576, -28672, 52352, 80256, 71936, 69632, 4864, -15232, 6528, -61696, 13824, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, -128, 127, 1175859562, 1325308050, 1807069200, 1420124271, 1188850090, 1182411826, 1831798253, 1649441596, 1897229520, 1960284024, 1088449917, 1672116795, 1343192385, 1487293887, 1672031270, 1674544455, 2061041943, 1951124498, 1660944822, 1092926841, 1362949302, 1686277765, 1889942367, 2094598643, 1884567220, 2137666356, 1554431692, 1602323645, 1300969388, 1742212516, 2087562224, 1697886333, 1743893709, 1139931211, 1408896915, 2007149259, 2008015774, 2115739605, 1364939324, 1669828999, 1224912049, 1117851976, 1283481598, 1789833187, 1167721435, 1534981936, 1382726870, 1907838383, 1372677674, 1120008250, 1414728578, 1400938390, 2122221780, 1317835141, 2026088264, 1627366011, 1688802527, 1657532372, 1082442094, 1658124163, 1837593306, 1539460112, 1223606748, 1717328997, -6, -7, -7, -7, -7, -6, -8, -7, -8, -7, -6, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7, -8, -7, -8, -8, -7, -7, -6, -7, -8, -7, -7, -7, -7, -8, -7, -7, -7, -8, -7, -7, -7, -8, -7, -7, -7, -7, -7, -6, -7, -6, -8, -7, -7, -7, -7, -7, -7, -7, -7, -7, -6, -7, 19200, 62592, -4352, 9600, -34560, 8576, -61568, -58240, 73856, 43264, 17024, -44160, 56448, 15872, -42496, -50432, -22272, -53632, -81664, -29312, -54016, 43648, 71552, 14336, 62592, -81024, -33024, 49792, 21760, 70272, -56448, 36480, 27648, -68992, 64000, -76160, 9856, 52480, 16768, 59648, 30720, -82176, 60544, 65536, 18048, 23680, 55424, 2048, -33408, 34304, 46464, 40192, 71168, 61184, 2560, 26752, -35200, -52864, 15872, -9344, 37120, -24448, -45440, 57088, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -128, 127, 1297315277, 1468492804, 1365389636, 1663912971, 1704283098, 1164680631, 1425183295, 1308370971, 1174422468, 1814191572, 1255544155, 1354838867, 1430538958, 1276332331, 1133893032, 1279299779, 2043925795, 1886366899, 1696393543, 2119679879, 1109649323, 1439202161, 1530326033, 1780452920, 1474144276, 1348916801, 2127510807, 2145988373, 1426113936, 1541155860, 2080786304, 1606089206, 1100905575, 1121256541, 2035983137, 1855576717, 1170383424, 2131232836, 1324947161, 1291685634, 1182123667, 1225577160, 1186540670, 1241111246, 1696464823, 1385422073, 1916933799, 1455092911, 1708877586, 1446115277, 1186109251, 1075221221, 1187914401, 1355229746, 1352046499, 1192666862, 1950388404, 1528773212, 1278521587, 1628880777, 1189265768, 1137548772, 1693155501, 1237104188, -7, -8, -7, -7, -7, -7, -7, -7, -7, -8, -7, -7, -7, -7, -7, -7, -8, -8, -8, -8, -7, -7, -7, -8, -7, -8, -8, -8, -7, -7, -8, -7, -7, -7, -8, -8, -7, -8, -7, -7, -7, -7, -7, -7, -7, -7, -8, -7, -8, -7, -7, -7, -7, -7, -7, -7, -8, -7, -7, -7, -7, -7, -8, -7, -49536, 95744, 17280, -23424, -23040, -26112, -41344, -33536, -46080, -66304, -88320, 31872, 16256, -384, 16512, -33280, -896, -83200, -52224, -33792, -41344, -17152, -18304, 123904, 19456, 163968, -115200, -26368, -43392, 3456, -51456, 53760, -75776, 9984, -57856, -49664, 22528, 37376, 15872, -116864, 15232, -73728, 14080, -40064, -40448, 45056, -61568, -28928, -108928, -57728, -48128, -51456, 17664, 35072, -23808, -22656, -67584, -59904, -19712, 66304, -70912, -45184, -72576, -63616, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, -128, 127, 1245499324, 2021708468, 1876208089, 1325188464, 1598729909, 1605918323, 1305633077, 2141131275, 1171759477, 1746719341, 2003695361, 1574723363, 2145427927, 1725225812, 1519686186, 2016259725, 2117951231, 1124575500, 2046616471, 1801639759, 1159154122, 1672063802, 1338164997, 1080975729, 1410880862, 1767309430, 1301111827, 1584185150, 1663224188, 1218672369, 1749809845, 1710495465, 1336342172, 1390854296, 1811826604, 1380530602, 1173781765, 2037710910, 1884565532, 2052846670, 1190845216, 1539715364, 1885930421, 1141038130, 1777548167, 1752818546, 1174930344, 1578116035, 1720287347, 1685584038, 1390238789, 1109472714, 1998071659, 1101556990, 1664121682, 1205680881, 1512822462, 1214391303, 1217392797, 1078342080, 1466779135, 1254490022, 1179852051, 1515358182, -7, -9, -8, -8, -8, -7, -7, -8, -6, -8, -7, -8, -8, -8, -7, -8, -8, -7, -8, -7, -7, -8, -8, -7, -7, -8, -7, -8, -7, -7, -7, -8, -7, -7, -7, -7, -7, -8, -7, -8, -7, -7, -8, -7, -8, -8, -6, -7, -7, -8, -6, -7, -8, -6, -7, -6, -7, -7, -7, -7, -7, -7, -6, -7, 57216, 87040, 87680, -71296, -84864, -36480, 67840, -82944, 19584, -80000, 72064, -89984, -69888, -98176, -11776, -86784, -65152, -78848, -93440, -2944, -83328, -69248, 92544, -55424, -44032, -71424, -86272, -72064, 4608, -29568, 70016, -54912, -72192, -45696, 43904, 10496, 87296, 76160, 86784, 96768, -95360, -25600, -68864, -67712, -85248, 75264, 87168, -48512, 92032, 87168, 6016, -87680, 61056, -22016, 57344, 21760, 73984, -69760, -80768, -62464, -71040, 24064, 84736, 83840, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -128, 127, 1288495350, 1272000227, 1133757147, 1325196533, 1544132175, 1576280665, 1250884107, 1416885595, 1257225235, 1398192069, 1182829336, 1322080746, 1344902500, 2033833852, 1719064480, 1135625170, 1170040216, 1353273889, 1554794931, 1202368357, 2132778438, 1528518266, 1492342135, 1710798759, 1149753748, 2079748779, 1262964489, 1636841300, 1329932912, 1678328183, 2045908555, 1181570288, 1473231915, 1208376190, 1989840828, 1382179700, 2082880260, 1915767688, 1245022969, 1320515753, 1168046573, 1200638585, 1970504475, 2076313148, 1433054747, 1128270331, 1241833722, 1370425141, 1955753547, 1321526997, 1258200908, 1358027456, 1304596498, 1414568521, 1282414133, 1231038246, 1459578170, 1460851567, 1551130029, 1218666866, 1131962883, 1482434001, 1846193434, 1482706018, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7, -8, -7, -7, -7, -7, -7, -7, -8, -7, -7, -7, -7, -8, -7, -8, -7, -7, -8, -7, -7, -7, -8, -7, -7, -8, -7, -7, -7, -7, -8, -8, -7, -7, -7, -7, -8, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7, -7, -8, -7, 23680, 1920, 50432, -1536, -57984, -16896, 31744, 5888, -5120, -63616, 88064, -17280, -20864, 35712, 12416, -51328, -19840, 17024, -27776, 21632, 640, -32384, 51200, -10496, 33024, 31104, 21248, 23808, 31872, -36992, -68224, 24960, 512, -1536, -26240, -48128, -22016, 2560, -18048, 256, -23680, -5248, 17664, 9216, -24960, -42368, -19328, 31104, -2560, 1280, -13184, 15360, 21376, 8192, -25472, 7168, 10624, -36096, -11904, -24832, 16896, -38400, 25600, -52864, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
};

int OpParamSummary(){
    printf("Operator parameter Summary:\n");
    printf("\tshared_param_avgpool: %d\n",sizeof(shared_param_avgpool));
    printf("\tshared_param_concat: %d\n",sizeof(shared_param_concat));
    printf("\tshared_param_conv: %d\n",sizeof(shared_param_conv));
    printf("\tshared_param_dwconv: %d\n",sizeof(shared_param_dwconv));
    printf("\tshared_param_fc: %d\n",sizeof(shared_param_fc));
    printf("\tshared_param_reshape: %d\n",sizeof(shared_param_reshape));
    printf("\tshared_param_softmax: %d\n",sizeof(shared_param_softmax));
    printf("\tshared_param_split: %d\n",sizeof(shared_param_split));
    printf("\top_data: %d\n", sizeof(op_data));
    return sizeof(shared_param_avgpool) + sizeof(shared_param_concat) + 
           sizeof(shared_param_conv) + sizeof(shared_param_dwconv) + 
           sizeof(shared_param_fc) + sizeof(shared_param_reshape) + 
           sizeof(shared_param_softmax) + sizeof(shared_param_split) +
           sizeof(op_data);
}