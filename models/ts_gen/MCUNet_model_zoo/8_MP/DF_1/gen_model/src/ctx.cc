#include "gen_model/include/ctx.h"
#include <cstdio>

alignas(16) int8_t arena[112288];
const int32_t arena_size = 112288;
const int32_t input_tid = 0;
const int32_t output_tid = 215;
const Tensor tensors[216] = {
    {{1, 176, 176, 3}, -1, 0, 0}, 
    {{16, 3, 3, 3}, 0, -1, 5}, 
    {{8, 1, 1, 16}, 432, -1, 5}, 
    {{24, 1, 1, 8}, 560, -1, 5}, 
    {{16, 1, 1, 24}, 752, -1, 5}, 
    {{48, 1, 1, 16}, 1136, -1, 5}, 
    {{16, 1, 1, 48}, 1904, -1, 5}, 
    {{48, 1, 1, 16}, 2672, -1, 5}, 
    {{16, 1, 1, 48}, 3440, -1, 5}, 
    {{48, 1, 1, 16}, 4208, -1, 5}, 
    {{16, 1, 1, 48}, 4976, -1, 5}, 
    {{48, 1, 1, 16}, 5744, -1, 5}, 
    {{16, 1, 1, 48}, 6512, -1, 5}, 
    {{48, 1, 1, 16}, 7280, -1, 5}, 
    {{16, 1, 1, 48}, 8048, -1, 5}, 
    {{96, 1, 1, 16}, 8816, -1, 5}, 
    {{24, 1, 1, 96}, 10352, -1, 5}, 
    {{72, 1, 1, 24}, 12656, -1, 5}, 
    {{24, 1, 1, 72}, 14384, -1, 5}, 
    {{72, 1, 1, 24}, 16112, -1, 5}, 
    {{24, 1, 1, 72}, 17840, -1, 5}, 
    {{72, 1, 1, 24}, 19568, -1, 5}, 
    {{24, 1, 1, 72}, 21296, -1, 5}, 
    {{144, 1, 1, 24}, 23024, -1, 5}, 
    {{32, 1, 1, 144}, 26480, -1, 5}, 
    {{96, 1, 1, 32}, 31088, -1, 5}, 
    {{32, 1, 1, 96}, 34160, -1, 5}, 
    {{96, 1, 1, 32}, 37232, -1, 5}, 
    {{32, 1, 1, 96}, 40304, -1, 5}, 
    {{96, 1, 1, 32}, 43376, -1, 5}, 
    {{32, 1, 1, 96}, 46448, -1, 5}, 
    {{192, 1, 1, 32}, 49520, -1, 5}, 
    {{64, 1, 1, 192}, 55664, -1, 5}, 
    {{384, 1, 1, 64}, 67952, -1, 5}, 
    {{64, 1, 1, 384}, 92528, -1, 5}, 
    {{192, 1, 1, 64}, 117104, -1, 5}, 
    {{64, 1, 1, 192}, 129392, -1, 5}, 
    {{192, 1, 1, 64}, 141680, -1, 5}, 
    {{64, 1, 1, 192}, 153968, -1, 5}, 
    {{384, 1, 1, 64}, 166256, -1, 5}, 
    {{96, 1, 1, 384}, 190832, -1, 5}, 
    {{384, 1, 1, 96}, 227696, -1, 5}, 
    {{1000, 1, 1, 384}, 264560, -1, 5}, 
    {{1, 16, 1, 1}, 648560, -1, -1}, 
    {{1, 8, 1, 1}, 648624, -1, -1}, 
    {{1, 24, 1, 1}, 648656, -1, -1}, 
    {{1, 16, 1, 1}, 648752, -1, -1}, 
    {{1, 48, 1, 1}, 648816, -1, -1}, 
    {{1, 16, 1, 1}, 649008, -1, -1}, 
    {{1, 48, 1, 1}, 649072, -1, -1}, 
    {{1, 16, 1, 1}, 649264, -1, -1}, 
    {{1, 48, 1, 1}, 649328, -1, -1}, 
    {{1, 16, 1, 1}, 649520, -1, -1}, 
    {{1, 48, 1, 1}, 649584, -1, -1}, 
    {{1, 16, 1, 1}, 649776, -1, -1}, 
    {{1, 48, 1, 1}, 649840, -1, -1}, 
    {{1, 16, 1, 1}, 650032, -1, -1}, 
    {{1, 96, 1, 1}, 650096, -1, -1}, 
    {{1, 24, 1, 1}, 650480, -1, -1}, 
    {{1, 72, 1, 1}, 650576, -1, -1}, 
    {{1, 24, 1, 1}, 650864, -1, -1}, 
    {{1, 72, 1, 1}, 650960, -1, -1}, 
    {{1, 24, 1, 1}, 651248, -1, -1}, 
    {{1, 72, 1, 1}, 651344, -1, -1}, 
    {{1, 24, 1, 1}, 651632, -1, -1}, 
    {{1, 144, 1, 1}, 651728, -1, -1}, 
    {{1, 32, 1, 1}, 652304, -1, -1}, 
    {{1, 96, 1, 1}, 652432, -1, -1}, 
    {{1, 32, 1, 1}, 652816, -1, -1}, 
    {{1, 96, 1, 1}, 652944, -1, -1}, 
    {{1, 32, 1, 1}, 653328, -1, -1}, 
    {{1, 96, 1, 1}, 653456, -1, -1}, 
    {{1, 32, 1, 1}, 653840, -1, -1}, 
    {{1, 192, 1, 1}, 653968, -1, -1}, 
    {{1, 64, 1, 1}, 654736, -1, -1}, 
    {{1, 384, 1, 1}, 654992, -1, -1}, 
    {{1, 64, 1, 1}, 656528, -1, -1}, 
    {{1, 192, 1, 1}, 656784, -1, -1}, 
    {{1, 64, 1, 1}, 657552, -1, -1}, 
    {{1, 192, 1, 1}, 657808, -1, -1}, 
    {{1, 64, 1, 1}, 658576, -1, -1}, 
    {{1, 384, 1, 1}, 658832, -1, -1}, 
    {{1, 96, 1, 1}, 660368, -1, -1}, 
    {{1, 384, 1, 1}, 660752, -1, -1}, 
    {{1, 1000, 1, 1}, 662288, -1, -1}, 
    {{1, 2, 1, 1}, 666288, -1, 1005}, 
    {{1, 2, 1, 1}, 666304, -1, 1005}, 
    {{1, 2, 1, 1}, 666320, -1, 1005}, 
    {{1, 1, 1, 1}, 666336, -1, 1005}, 
    {{1, 176, 176, 3}, -176, 0, 1005}, 
    {{1, 88, 88, 16}, -88, 176, 1006}, 
    {{1, 3, 3, 16}, 666352, -1, 5}, 
    {{1, 16, 1, 1}, 666496, -1, -1}, 
    {{1, 88, 88, 16}, -88, 264, 1007}, 
    {{1, 88, 88, 8}, -88, 352, 1008}, 
    {{1, 88, 88, 24}, -88, 440, 1009}, 
    {{1, 5, 5, 24}, 666560, -1, 5}, 
    {{1, 24, 1, 1}, 667168, -1, -1}, 
    {{1, 2, 1, 1}, 667264, -1, 1005}, 
    {{1, 44, 44, 24}, -44, 528, 1010}, 
    {{1, 44, 44, 16}, -44, 572, 1011}, 
    {{1, 44, 44, 48}, -44, 616, 1012}, 
    {{1, 2, 1, 1}, 667280, -1, 1005}, 
    {{1, 3, 3, 48}, 667296, -1, 5}, 
    {{1, 48, 1, 1}, 667728, -1, -1}, 
    {{1, 44, 44, 48}, -44, 660, 1013}, 
    {{1, 44, 44, 16}, -44, 704, 1014}, 
    {{1, 44, 44, 16}, -44, 748, 1015}, 
    {{1, 44, 44, 48}, -44, 792, 1016}, 
    {{1, 7, 7, 48}, 667920, -1, 5}, 
    {{1, 48, 1, 1}, 670272, -1, -1}, 
    {{1, 2, 1, 1}, 670464, -1, 1005}, 
    {{1, 22, 22, 48}, -22, 836, 1017}, 
    {{1, 22, 22, 16}, -22, 858, 1018}, 
    {{1, 22, 22, 48}, -22, 880, 1019}, 
    {{1, 2, 1, 1}, 670480, -1, 1005}, 
    {{1, 3, 3, 48}, 670496, -1, 5}, 
    {{1, 48, 1, 1}, 670928, -1, -1}, 
    {{1, 22, 22, 48}, -22, 902, 1020}, 
    {{1, 22, 22, 16}, -22, 924, 1021}, 
    {{1, 22, 22, 16}, -22, 946, 5}, 
    {{1, 22, 22, 48}, -22, 968, 1022}, 
    {{1, 2, 1, 1}, 671120, -1, 1005}, 
    {{1, 5, 5, 48}, 671136, -1, 5}, 
    {{1, 48, 1, 1}, 672336, -1, -1}, 
    {{1, 22, 22, 48}, -22, 990, 1023}, 
    {{1, 22, 22, 16}, -22, 1012, 1024}, 
    {{1, 22, 22, 16}, -22, 1034, 1025}, 
    {{1, 22, 22, 48}, -22, 1056, 1026}, 
    {{1, 2, 1, 1}, 672528, -1, 1005}, 
    {{1, 5, 5, 48}, 672544, -1, 5}, 
    {{1, 48, 1, 1}, 673744, -1, -1}, 
    {{1, 22, 22, 48}, -22, 1078, 1027}, 
    {{1, 22, 22, 16}, -22, 1100, 1028}, 
    {{1, 22, 22, 16}, -22, 1122, 1029}, 
    {{1, 22, 22, 96}, -22, 1144, 1030}, 
    {{1, 7, 7, 96}, 673936, -1, 5}, 
    {{1, 96, 1, 1}, 678640, -1, -1}, 
    {{1, 11, 11, 96}, -11, 1166, 1031}, 
    {{1, 11, 11, 24}, -11, 1177, 1032}, 
    {{1, 11, 11, 72}, -11, 1188, 1033}, 
    {{1, 5, 5, 72}, 679024, -1, 5}, 
    {{1, 72, 1, 1}, 680832, -1, -1}, 
    {{1, 11, 11, 72}, -11, 1199, 1034}, 
    {{1, 11, 11, 24}, -11, 1210, 1035}, 
    {{1, 11, 11, 24}, -11, 1221, 1036}, 
    {{1, 11, 11, 72}, -11, 1232, 1037}, 
    {{1, 5, 5, 72}, 681120, -1, 5}, 
    {{1, 72, 1, 1}, 682928, -1, -1}, 
    {{1, 11, 11, 72}, -11, 1243, 1038}, 
    {{1, 11, 11, 24}, -11, 1254, 1039}, 
    {{1, 11, 11, 24}, -11, 1265, 1040}, 
    {{1, 11, 11, 72}, -11, 1276, 1041}, 
    {{1, 5, 5, 72}, 683216, -1, 5}, 
    {{1, 72, 1, 1}, 685024, -1, -1}, 
    {{1, 11, 11, 72}, -11, 1287, 1042}, 
    {{1, 11, 11, 24}, -11, 1298, 1043}, 
    {{1, 11, 11, 24}, -11, 1309, 1044}, 
    {{1, 11, 11, 144}, -11, 1320, 1045}, 
    {{1, 5, 5, 144}, 685312, -1, 5}, 
    {{1, 144, 1, 1}, 688912, -1, -1}, 
    {{1, 11, 11, 144}, -11, 1331, 1046}, 
    {{1, 11, 11, 32}, -11, 1342, 1047}, 
    {{1, 11, 11, 96}, -11, 1353, 1048}, 
    {{1, 5, 5, 96}, 689488, -1, 5}, 
    {{1, 96, 1, 1}, 691888, -1, -1}, 
    {{1, 11, 11, 96}, -11, 1364, 1049}, 
    {{1, 11, 11, 32}, -11, 1375, 1050}, 
    {{1, 11, 11, 32}, -11, 1386, 1051}, 
    {{1, 11, 11, 96}, -11, 1397, 1052}, 
    {{1, 5, 5, 96}, 692272, -1, 5}, 
    {{1, 96, 1, 1}, 694672, -1, -1}, 
    {{1, 11, 11, 96}, -11, 1408, 1053}, 
    {{1, 11, 11, 32}, -11, 1419, 1054}, 
    {{1, 11, 11, 32}, -11, 1430, 1055}, 
    {{1, 11, 11, 96}, -11, 1441, 1056}, 
    {{1, 5, 5, 96}, 695056, -1, 5}, 
    {{1, 96, 1, 1}, 697456, -1, -1}, 
    {{1, 11, 11, 96}, -11, 1452, 1057}, 
    {{1, 11, 11, 32}, -11, 1463, 1058}, 
    {{1, 11, 11, 32}, -11, 1474, 1059}, 
    {{1, 11, 11, 192}, -11, 1485, 1060}, 
    {{1, 7, 7, 192}, 697840, -1, 5}, 
    {{1, 192, 1, 1}, 707248, -1, -1}, 
    {{1, 6, 6, 192}, -6, 1496, 1061}, 
    {{1, 6, 6, 64}, -6, 1502, 1062}, 
    {{1, 6, 6, 384}, -6, 1508, 1063}, 
    {{1, 7, 7, 384}, 708016, -1, 5}, 
    {{1, 384, 1, 1}, 726832, -1, -1}, 
    {{1, 6, 6, 384}, -6, 1514, 1064}, 
    {{1, 6, 6, 64}, -6, 1520, 1065}, 
    {{1, 6, 6, 64}, -6, 1526, 1066}, 
    {{1, 6, 6, 192}, -6, 1532, 1067}, 
    {{1, 2, 1, 1}, 728368, -1, 1005}, 
    {{1, 7, 7, 192}, 728384, -1, 5}, 
    {{1, 192, 1, 1}, 737792, -1, -1}, 
    {{1, 6, 6, 192}, -6, 1538, 1068}, 
    {{1, 6, 6, 64}, -6, 1544, 1069}, 
    {{1, 6, 6, 64}, -6, 1550, 1070}, 
    {{1, 6, 6, 192}, -6, 1556, 1071}, 
    {{1, 7, 7, 192}, 738560, -1, 5}, 
    {{1, 192, 1, 1}, 747968, -1, -1}, 
    {{1, 6, 6, 192}, -6, 1562, 1072}, 
    {{1, 6, 6, 64}, -6, 1568, 1073}, 
    {{1, 6, 6, 64}, -6, 1574, 1074}, 
    {{1, 6, 6, 384}, -6, 1580, 1075}, 
    {{1, 7, 7, 384}, 748736, -1, 5}, 
    {{1, 384, 1, 1}, 767552, -1, -1}, 
    {{1, 6, 6, 384}, -6, 1586, 1076}, 
    {{1, 6, 6, 96}, -6, 1592, 1077}, 
    {{1, 6, 6, 384}, -6, 1598, 1078}, 
    {{1, 6, 6, 384}, -1, 0, 1}, 
    {{1, 1, 1, 384}, -1, 13824, 2}, 
    {{1, 1, 1, 1000}, -1, 0, 3}, 
    {{1, 2, 1, 1}, 769088, -1, 1005}, 
    {{1, 1000, 1, 1}, -1, 1008, 4}
};
const int all_0_zp_cursor = 5;
const int only_zp_cursor = 1005;
const int only_zp_start = 1005;
const int32_t quant_scale[1005] = {
    1006665857, 1019265217, 1019265217, 1043002604, 1043002604, 992995223, 995996622, 993269354, 992122905, 995432271, 992063734, 994947832, 992901646, 992000627, 998213351, 992355627, 993718669, 993844973, 995299856, 994307732, 993131803, 991498751, 996316044, 993497233, 996040454, 997211788, 992352409, 993161990, 997379684, 992882544, 992922464, 991630755, 992515640, 992936049, 996054843, 994458057, 995724328, 993018637, 992257736, 998059424, 992832793, 992736332, 991537234, 995728337, 994440162, 993524329, 993508691, 995511537, 995954190, 995162864, 993584109, 993664568, 993871407, 993498581, 995166894, 995079514, 993397496, 992697007, 993694770, 992389126, 995465095, 995569173, 995120310, 996326873, 994632405, 992293901, 996935122, 992870303, 997963076, 992401841, 997012968, 993718763, 995172070, 993908822, 996611556, 996130360, 994942752, 993936515, 992295353, 993943448, 993095881, 993951401, 994470403, 996883206, 994671021, 993676650, 998969855, 996323627, 993446551, 997709111, 992311287, 992401021, 995961585, 997117526, 994551767, 994199182, 992923542, 994239231, 992558139, 994095356, 993822032, 993490623, 996421571, 995185245, 994290866, 998112288, 993569501, 994269258, 993779576, 997463414, 995328731, 994166995, 994865664, 993444875, 992348228, 999091276, 995159525, 993326744, 991378690, 992215854, 995251671, 993696031, 992863162, 997053086, 992808722, 995881976, 993930142, 1001306830, 994461497, 998757327, 994388655, 998053849, 996035113, 995068855, 997503470, 993724871, 995659618, 995372266, 994444665, 997495469, 995003218, 992876713, 996129839, 996353606, 998107221, 993360471, 994979706, 994809228, 992231109, 993313470, 995824622, 992602930, 993313860, 995380619, 994739765, 995235680, 990875741, 992218598, 994384183, 992204805, 994812717, 992267996, 995084624, 992257330, 993639048, 992109579, 994367660, 991773241, 993060390, 993120750, 992154109, 991706038, 993795721, 991071324, 998409353, 992001785, 991961488, 993041077, 990938395, 991965229, 992839151, 994755211, 991984095, 992037589, 993124063, 992691982, 993480760, 994556733, 991299688, 990891168, 993044869, 993727795, 993553758, 992783153, 994031006, 995257991, 997592805, 992482385, 992583190, 993204502, 993716727, 995602355, 989901007, 993562742, 993268247, 991048002, 992474271, 991469703, 994532084, 992402703, 993135433, 991441149, 991899755, 996986882, 992230273, 994955364, 993817044, 994521710, 994792087, 994965686, 993957435, 993606909, 991832836, 993976923, 994693288, 992876130, 991705822, 990873662, 995566266, 990357406, 995832834, 996043959, 993026580, 995947639, 994440231, 993721402, 996373220, 993940528, 992971027, 991423095, 996616730, 990049532, 997857725, 996387333, 994514696, 993185391, 997249399, 992238972, 992864672, 992249577, 993205992, 994725667, 994094156, 993513519, 994608434, 992124263, 991448899, 994124722, 993791406, 995600263, 993102734, 994685688, 990121353, 992940486, 992108396, 994465601, 993236718, 992424207, 992200341, 993306731, 994278013, 995985136, 994529898, 990576715, 991431010, 991769752, 1002505132, 991516565, 993682749, 991377660, 993138369, 993948831, 994638106, 993022897, 993154061, 992154967, 996997492, 995342307, 992506329, 992190651, 994423692, 993049366, 995406711, 991818727, 996099609, 997443996, 996295434, 992635729, 994051825, 996351146, 994669532, 997423707, 994277762, 995678831, 993586597, 996218700, 997077578, 993084190, 993897881, 995616218, 993926999, 998554779, 996202099, 994828019, 994937448, 993918055, 993914994, 995825039, 996016802, 992530674, 995929077, 994022631, 994410623, 994148917, 994634829, 993421935, 994118600, 995077733, 992133435, 992686133, 992752814, 993624238, 994841458, 995570258, 990940476, 992531138, 994205734, 991549759, 991529054, 993739481, 993884998, 991087068, 995100574, 991696078, 996409009, 992283927, 991552305, 991768602, 994142900, 995176756, 995540878, 995082314, 993297709, 996723243, 994983472, 992982994, 992596802, 991184711, 992775630, 994410122, 994586921, 991140561, 992111156, 993623721, 996606865, 996119715, 993121957, 991562482, 991484051, 993708366, 993739848, 993139272, 994155126, 995698236, 991847682, 994158103, 992772274, 993445023, 997330313, 995048490, 994302168, 995861386, 992124964, 995198098, 996997006, 993869259, 992944239, 990911139, 995167786, 995723211, 992523818, 994260844, 992340678, 995546660, 997964264, 994427554, 996678289, 998042822, 995486055, 994503075, 997249594, 993802415, 996744627, 995113165, 996717815, 995261308, 993693057, 995658347, 998777798, 996419537, 992991495, 991463806, 992175376, 1001560730, 991859256, 997369721, 997040817, 991933925, 994108384, 991626412, 993260966, 995940205, 992398234, 993131768, 992616838, 993358289, 989505637, 994207463, 993840397, 1000964029, 993185082, 993939233, 994963693, 991515305, 1002514596, 995059318, 992934224, 994977603, 994975261, 995189196, 994146263, 996523152, 993426861, 992115722, 998565718, 990577196, 992763588, 991844215, 993279706, 993845058, 996088010, 990483833, 992663391, 992256072, 999302122, 995026901, 995304034, 994290078, 991979033, 992589602, 996595067, 994617018, 992478316, 994052008, 991768497, 992941364, 993163772, 994839337, 991550075, 991024217, 991256388, 997050378, 992889414, 995200394, 994882223, 993907467, 994063199, 995203585, 992385103, 991961838, 998442789, 994905834, 991300414, 992454975, 997859312, 991447564, 991803164, 994606351, 994331472, 996053057, 993212522, 997809698, 994130393, 992392642, 995069800, 994134298, 1000141383, 990916990, 994861986, 995681749, 992868143, 993163126, 991406266, 994537754, 992080536, 993703380, 992903381, 994264116, 995082817, 994313006, 995381280, 997187227, 995125682, 993097425, 990260054, 993992144, 996824277, 993666807, 993102068, 993523972, 991453167, 992161261, 993143444, 998617521, 993780652, 994671715, 992816074, 998931191, 992325083, 991923393, 996878615, 999907523, 991376753, 993331292, 993108493, 991242980, 993305904, 993644803, 994755834, 1000617693, 994625690, 997038189, 998619877, 990956220, 991957610, 997792456, 990300136, 993574014, 995629666, 993320732, 992607997, 996852750, 993072797, 995027425, 990807633, 998690372, 993814662, 993621790, 995296905, 998743652, 994130021, 991952990, 993529133, 995503665, 993495627, 1000517032, 994223904, 991714655, 993662793, 999606845, 993164212, 991724636, 994837458, 993077241, 995379486, 994975682, 992430823, 991037819, 993699198, 993758510, 993755905, 999360894, 990537452, 992919050, 992671358, 993097932, 994307693, 989313243, 996522947, 1001470840, 995197890, 998578799, 992088474, 994519255, 994585255, 992165601, 991341971, 994566460, 994264653, 995681663, 995669122, 996448562, 992819710, 993908263, 992466893, 990582429, 995336166, 1000265630, 993133664, 995765992, 998486887, 998274715, 991768529, 994598083, 993841933, 995163934, 997090526, 990454579, 990610462, 993700121, 992396116, 992773002, 992156548, 992908378, 990551793, 992815165, 993313518, 994628058, 991606952, 993147783, 996782761, 991263193, 994572678, 994774308, 998884904, 998882457, 992206100, 993163746, 992323174, 992758090, 991494091, 992058494, 992283482, 995458479, 998468613, 994883322, 992331263, 992554552, 994343598, 996924848, 994553813, 997612186, 992497029, 996505739, 990662242, 991495393, 994005485, 996462994, 993573166, 994817624, 994202161, 995312774, 997026332, 993214956, 992381189, 991144174, 993498989, 991061902, 994413768, 995816885, 991929632, 991110733, 995542258, 995513559, 991981841, 994109670, 994622355, 992828771, 995865796, 993281257, 996916474, 992421458, 997591068, 993508151, 993759842, 992907102, 992536106, 995933182, 997326733, 998310009, 998470765, 995214454, 996148410, 996883900, 997838940, 994172894, 994723405, 988517844, 994402052, 1000058759, 996682066, 992092132, 993179855, 999966884, 994174361, 993030357, 994112948, 996513134, 990214006, 997196529, 995007518, 994344113, 993444542, 994143192, 990818953, 994694374, 990432904, 993366044, 990102938, 992693670, 996403655, 992602245, 994692964, 991668717, 992120236, 992523227, 991511803, 996709403, 992343409, 998761665, 993504838, 992422096, 992287831, 992480438, 993785507, 995273413, 991083274, 992823575, 991791837, 992143444, 995320456, 994117434, 992091307, 992198833, 993619971, 991182389, 992140890, 989242533, 997093308, 995980170, 993490196, 1000926249, 995847908, 991297906, 994739335, 991805305, 993714524, 994616119, 995462235, 991119421, 994178744, 990942461, 992219342, 995046149, 993348129, 998816193, 994251786, 993268678, 994394718, 994443301, 992342822, 993414257, 995125346, 994721433, 994123331, 993916906, 995675351, 995450094, 992130721, 994697260, 991646980, 993370941, 994627930, 993705598, 998017492, 993606705, 995290716, 994008309, 997051684, 991456964, 996461719, 996277444, 997477279, 998520992, 992441494, 996257290, 995227950, 995463381, 992083428, 992826301, 994881158, 993447533, 991552800, 993606482, 989313630, 995621291, 994716295, 995006692, 995566376, 996532547, 997854095, 992885691, 994459193, 992688223, 992538536, 993882282, 992592147, 999753507, 993107975, 994665093, 990414994, 997186689, 999064804, 995654914, 992342897, 995708433, 995510183, 993285544, 994658925, 993022179, 992002494, 996473628, 994495033, 992479492, 993131083, 994232917, 990422358, 994424805, 994731890, 991756876, 992699424, 997198142, 995066898, 994721963, 992321500, 995320319, 994185723, 1000637581, 995001466, 992284753, 997115550, 992205979, 999813263, 990664205, 992953375, 992401503, 994285534, 995360059, 993044725, 997659420, 994408136, 995403029, 994787816, 992654982, 992904232, 992240835, 993630503, 995981050, 994149685, 992690237, 990332865, 993997702, 992971867, 996026061, 993124043, 991666644, 997327233, 996087517, 997410507, 993247696, 992519881, 992085817, 994411449, 993690940, 993930979, 991336385, 992755748, 992354560, 990925663, 992539977, 993424307, 990843254, 995098367, 997686972, 997293138, 1000527522, 996574029, 993501552, 991799928, 993423264, 994869879, 995066465, 993720141, 991826904, 993247817, 995302483, 992308919, 992955591, 992392810, 995504306, 995653257, 995088295, 992567520, 995513272, 992953124, 992926150, 991404635, 995253979, 993813372, 995700780, 994738011, 998653686, 997231737, 991785751, 992306866, 993439064, 998365131, 997027320, 990519796, 995824436, 994858062, 993516306, 997694364, 993045716, 994505992, 991068145, 992722027, 995070449, 997741679, 992309300, 994894650, 996007256, 993830069, 995067750, 997300645, 997353053, 999007487, 994503277, 997443049, 995821245, 992626083, 991386948, 992245983, 995283124, 992972320, 992283662, 997996892, 994871124, 997180339, 993681912, 998829814, 996562815, 999169074, 997429011, 993244082, 993223952, 990910407, 992711283, 995664067, 993569766, 992873358, 995927505, 992340606, 994311301, 993800624, 990659641, 994289930, 996556907, 991908236, 996296377, 995574647, 994219457, 992869571, 994032190, 991404441, 995800620, 997899118, 994487068, 992800683, 997851327, 993537141, 994515322, 992013514, 993131838, 997558702, 995187932, 994152420, 993045409, 996424088, 993645812, 993188525, 996381196, 994704001, 996021153, 994245115
};
const int8_t quant_zeropoint[1079] = {
    -1, -128, -128, -36, -36, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -128, -128, 1, -128, -128, 22, -128, -128, -16, -4, -128, -128, -7, -128, -128, 40, -128, -128, -4, 2, -128, -128, 18, 1, -128, -128, 3, -128, -128, -23, -7, -128, -128, -31, -10, -128, -128, -5, 7, -128, -128, -6, -128, -128, 34, -3, -128, -128, 5, -4, -128, -128, 18, 1, -128, -128, -6, -128, -128, 9, 6, -128, -128, -12, -11, -128, -128, -9, -30, -128, -128, 5, -128
};
const int32_t split_offset[1604] = {
    0, 528, 1056, 1584, 2112, 2640, 3168, 3696, 4224, 4752, 5280, 5808, 6336, 6864, 7392, 7920, 8448, 8976, 9504, 10032, 10560, 11088, 11616, 12144, 12672, 13200, 13728, 14256, 14784, 15312, 15840, 16368, 16896, 17424, 17952, 18480, 19008, 19536, 20064, 20592, 21120, 21648, 22176, 22704, 23232, 23760, 24288, 24816, 25344, 25872, 26400, 26928, 27456, 27984, 28512, 29040, 29568, 30096, 30624, 31152, 31680, 32208, 32736, 33264, 33792, 34320, 34848, 35376, 35904, 36432, 36960, 37488, 38016, 38544, 39072, 39600, 40128, 40656, 41184, 41712, 42240, 42768, 43296, 43824, 44352, 44880, 45408, 45936, 46464, 46992, 47520, 48048, 48576, 49104, 49632, 50160, 50688, 51216, 51744, 52272, 52800, 53328, 53856, 54384, 54912, 55440, 55968, 56496, 57024, 57552, 58080, 58608, 59136, 59664, 60192, 60720, 61248, 61776, 62304, 62832, 63360, 63888, 64416, 64944, 65472, 66000, 66528, 67056, 67584, 68112, 68640, 69168, 69696, 70224, 70752, 71280, 71808, 72336, 72864, 73392, 73920, 74448, 74976, 75504, 76032, 76560, 77088, 77616, 78144, 78672, 79200, 79728, 80256, 80784, 81312, 81840, 82368, 82896, 83424, 83952, 84480, 85008, 85536, 86064, 86592, 87120, 87648, 88176, 88704, 89232, 89760, 90288, 90816, 91344, 91872, 92400, 95040, 96448, 99264, 100672, 102080, 103488, 104896, 106304, 103488, 104896, 106304, 103488, 104896, 106304, 103488, 104896, 106304, 103488, 101376, 104896, 95040, 96448, 97856, 101376, 23232, 95040, 96448, 97856, 25344, 26752, 28160, 29568, 30976, 27456, 29568, 30976, 32384, 29568, 33792, 31680, 35200, 33088, 36608, 34496, 38016, 35904, 39424, 37312, 35904, 38720, 40128, 41536, 33792, 38016, 39424, 40832, 38016, 39424, 40832, 42240, 33792, 40128, 41536, 42944, 38016, 39424, 40832, 42240, 33792, 40128, 41536, 42944, 38016, 39424, 40832, 42240, 33792, 40128, 41536, 42944, 38016, 39424, 40832, 42240, 33792, 40128, 41536, 42944, 0, 0, 2112, 2112, 4224, 4224, 6336, 6336, 8448, 8448, 10560, 10560, 12672, 12672, 14784, 14784, 16896, 16896, 19008, 0, 19008, 21120, 23232, 2112, 4224, 21120, 23232, 6336, 8448, 21120, 25344, 12672, 23232, 21120, 32384, 16896, 25344, 23232, 35200, 19008, 27456, 25344, 31680, 4224, 29568, 27456, 33792, 8448, 31680, 29568, 35904, 10560, 21120, 31680, 33792, 23232, 35904, 31680, 38016, 0, 14784, 31680, 33792, 27456, 35904, 31680, 38016, 2112, 6336, 31680, 33792, 21120, 35904, 31680, 38016, 12672, 16896, 31680, 33792, 14784, 35904, 31680, 38016, 4224, 19008, 31680, 33792, 6336, 97856, 95040, 97152, 99264, 101376, 101376, 103488, 97152, 106304, 99264, 104896, 101376, 103488, 97152, 106304, 92928, 99264, 99264, 103488, 19008, 21120, 92928, 95040, 4224, 24640, 23232, 25344, 8448, 28160, 25344, 27456, 23232, 28864, 28864, 34496, 25344, 30976, 30976, 33088, 27456, 31680, 31680, 33792, 29568, 33792, 33792, 35904, 31680, 37312, 35904, 38016, 21120, 35200, 33792, 38016, 35904, 40832, 38016, 40128, 14784, 25344, 33792, 35904, 35904, 40832, 38016, 40128, 6336, 29568, 33792, 35904, 35904, 40832, 38016, 40128, 16896, 25344, 33792, 35904, 35904, 40832, 38016, 40128, 19008, 29568, 33792, 35904, 39072, 92928, 0, 95040, 2112, 99264, 4224, 101376, 6336, 97152, 8448, 99264, 10560, 101376, 12672, 97152, 14784, 92928, 16896, 99264, 0, 19008, 21120, 92928, 2112, 4224, 21120, 23232, 6336, 8448, 21120, 25344, 12672, 23232, 21120, 27456, 16896, 25344, 23232, 29568, 19008, 27456, 25344, 31680, 4224, 29568, 27456, 33792, 8448, 31680, 29568, 35904, 10560, 21120, 31680, 33792, 23232, 35904, 31680, 38016, 0, 14784, 31680, 33792, 27456, 35904, 31680, 38016, 2112, 6336, 31680, 33792, 21120, 35904, 31680, 38016, 12672, 16896, 31680, 33792, 14784, 35904, 31680, 38016, 4224, 19008, 31680, 33792, 6336, 2112, 4224, 6336, 8448, 10560, 12672, 14784, 16896, 19008, 21120, 23232, 24640, 25344, 28160, 30976, 28864, 32384, 30976, 35200, 31680, 38016, 33792, 35904, 37312, 38016, 39424, 38016, 40832, 53504, 25344, 35904, 40832, 43648, 29568, 35904, 40832, 52448, 25344, 35904, 40832, 45408, 29568, 35904, 39072, 104896, 107712, 108416, 107712, 108768, 109472, 110880, 102784, 106304, 104896, 97152, 101376, 102080, 96096, 32384, 33792, 34496, 36608, 37312, 39072, 39776, 40832, 41536, 48224, 51392, 42240, 42944, 54560, 55616, 44352, 58272, 47168, 59872, 44352, 59872, 43648, 45408, 54560, 45408, 43648, 46464, 63872, 40128, 31680, 97152, 0, 2112, 4224, 6336, 8448, 10560, 12672, 14784, 16896, 0, 19008, 2112, 4224, 6336, 8448, 12672, 21120, 16896, 23232, 19008, 25344, 4224, 27456, 8448, 29568, 10560, 21120, 23232, 25344, 0, 14784, 27456, 29568, 2112, 6336, 21120, 25344, 12672, 16896, 14784, 29568, 4224, 6336, 92928, 95040, 99264, 101376, 97152, 99264, 99264, 97152, 92928, 95040, 21120, 25344, 21120, 23232, 21120, 25344, 23232, 27456, 25344, 29568, 27456, 31680, 29568, 33792, 31680, 35904, 31680, 33792, 31680, 35904, 31680, 33792, 31680, 35904, 31680, 33792, 31680, 35904, 31680, 33792, 31680, 35904, 19008, 29568, 4224, 6336, 0, 2112, 4224, 6336, 8448, 10560, 12672, 14784, 16896, 0, 19008, 2112, 4224, 6336, 8448, 12672, 21120, 16896, 23232, 19008, 25344, 4224, 27456, 8448, 29568, 10560, 21120, 23232, 25344, 0, 14784, 27456, 29568, 2112, 6336, 21120, 25344, 12672, 16896, 14784, 29568, 4224, 4928, 7040, 8448, 10560, 12672, 14784, 16896, 19008, 21120, 23232, 21120, 25344, 21120, 23232, 21120, 25344, 23232, 27456, 25344, 29568, 27456, 31680, 29568, 33792, 31680, 35904, 31680, 33792, 31680, 35904, 31680, 33792, 31680, 35904, 31680, 33792, 31680, 35904, 31680, 33792, 31680, 35904, 30272, 6336, 92928, 95040, 0, 2112, 4224, 6336, 8448, 10560, 12672, 14784, 16896, 0, 19008, 2112, 4224, 6336, 8448, 12672, 21120, 16896, 23232, 19008, 25344, 4224, 27456, 8448, 29568, 10560, 21120, 23232, 25344, 0, 14784, 27456, 29568, 2112, 6336, 21120, 25344, 12672, 16896, 14784, 19008, 4224, 10560, 14784, 19008, 23232, 25344, 23232, 25344, 27456, 29568, 31680, 33792, 35904, 33792, 35904, 33792, 35904, 33792, 35904, 33792, 35904, 6336, 8448, 110176, 111584, 107008, 105952, 101376, 102080, 106304, 40480, 42240, 49280, 52448, 55264, 58976, 44704, 61632, 60576, 63392, 62272, 65280, 40832, 22704, 11616, 107712, 108768, 102784, 99264, 92928, 96096, 99264, 102432, 42944, 45056, 48224, 51392, 53504, 55616, 47168, 50336, 52448, 54560, 45408, 35904, 25344, 8448, 14784, 0, 2112, 6336, 10560, 14784, 0, 2112, 6336, 12672, 16896, 19008, 4224, 8448, 10560, 23232, 0, 27456, 2112, 6336, 4224, 9504, 15840, 1056, 3168, 7392, 11616, 15840, 1056, 3168, 7392, 13728, 17952, 20064, 5280, 9504, 11616, 24288, 1056, 28512, 3168, 7392, 5280, 8448, 111936, 107360, 106304, 101728, 103488, 106656, 104192, 99264, 99968, 43648, 97856, 59328, 60928, 61984, 62688, 63744, 64576, 65632, 41184, 33024, 16368, 17424, 109824, 103840, 100320, 93984, 97152, 92928, 96096, 44000, 46112, 49280, 45056, 48224, 51392, 53504, 55616, 47168, 50336, 52448, 36960, 26400, 10560, 11616, 2112, 6336, 10560, 14784, 0, 2112, 6336, 12672, 16896, 19008, 4224, 8448, 10560, 23232, 0, 27456, 2112, 6336, 4224, 8448, 0, 16368, 3168, 7392, 11616, 15840, 1056, 3168, 7392, 13728, 17952, 20064, 5280, 9504, 11616, 24288, 1056, 28512, 3168, 7392, 5280, 9504, 1056, 10560, 106656, 102784, 105952, 103840, 104544, 99616, 97152, 97504, 95840, 60576, 61280, 62336, 63040, 64096, 64928, 65984, 46464, 33376, 16720, 20064, 33024, 33376, 104896, 95040, 98208, 101376, 100320, 93984, 47168, 50336, 52448, 54560, 46112, 49280, 44352, 48224, 51392, 53504, 38016, 29568, 12672, 13728, 17952, 19008, 10560, 14784, 0, 2112, 6336, 12672, 16896, 19008, 4224, 8448, 10560, 23232, 0, 27456, 2112, 6336, 4224, 8448, 0, 10560, 2112, 12672, 11616, 15840, 1056, 3168, 7392, 13728, 17952, 20064, 5280, 9504, 11616, 24288, 1056, 28512, 3168, 7392, 5280, 9504, 1056, 11616, 3168, 13728, 12672, 16896, 19008, 4224, 8448, 21120, 23232, 25344, 27456, 29568, 21120, 25344, 14784, 29568, 6336, 21120, 9504, 14784, 16368, 16368, 12672, 17952, 10560, 14784, 0, 2112, 6336, 12672, 16896, 19008, 4224, 8448, 10560, 23232, 0, 27456, 2112, 6336, 4224, 8448, 0, 10560, 2112, 12672, 4224, 21120, 25344, 29568, 25344, 29568, 21120, 14784, 16368, 17952, 5808, 103488, 98208, 93728, 66336, 66880, 67696, 33728, 34544, 29456, 19360, 9696, 95040, 92928, 56672, 58272, 59872, 49280, 30624, 22704, 25104, 7392, 8192, 0, 2112, 12672, 4224, 8448, 0, 2112, 4224, 0, 1584, 1584, 800, 2912, 13472, 5024, 9248, 800, 2912, 5024, 800, 2384, 2384, 94000, 66608, 67152, 67968, 34000, 34816, 29728, 19632, 9968, 21120, 24640, 57472, 59072, 60672, 62272, 31424, 23504, 25904, 27504, 10560, 29104, 23232, 12672, 4224, 8448, 0, 2112, 4224, 0, 1584, 1584, 1584, 1584, 13472, 5024, 9248, 800, 2912, 5024, 800, 2384, 2384, 2384, 2384, 67424, 66336, 34272, 35088, 30000, 33024, 10240, 21392, 24912, 26752, 18336, 61472, 63072, 32224, 24304, 26704, 28304, 11360, 29904, 30704, 31504, 32304, 8448, 0, 2112, 4224, 0, 1584, 1584, 1584, 1584, 0, 0, 9248, 800, 2912, 5024, 800, 2384, 2384, 2384, 2384, 800, 800, 8448, 0, 2112, 5808, 1584, 1584, 1584, 1584, 1584, 0, 0, 21120, 14784, 16368, 4224, 0, 5808, 7392, 8976, 10944, 12528, 14112, 2112, 5808, 1584, 1584, 1584, 1584, 1584, 0, 0, 0, 15696, 29104, 19008, 8992, 12160, 24288, 4416, 10560, 33104, 32064, 34160, 33120, 20064, 17952, 3168, 14112, 16224, 21120, 23232, 26400, 29568, 19392, 21504, 1584, 1584, 1584, 1584, 0, 0, 0, 0, 10944, 13056, 15168, 2640, 2640, 2640, 2640, 1056, 1056, 1056, 1056, 12000, 14112, 19392, 9344, 13584, 25344, 4768, 1584, 33456, 32416, 34512, 16320, 32064, 29760, 12528, 15168, 19008, 17280, 24288, 27456, 30624, 20448, 22560, 24672, 21504, 1584, 1584, 0, 0, 0, 0, 10944, 13056, 15168, 18336, 6912, 2640, 2640, 1056, 1056, 1056, 1056, 12000, 14112, 19392, 19392, 7968, 26400, 5120, 1936, 15696, 33808, 32768, 16672, 32416, 30112, 30464, 30816, 20064, 22176, 25344, 28512, 18336, 17280, 23616, 25728, 26880, 27936, 24672, 0, 0, 0, 0, 10944, 13056, 15168, 18336, 6912, 6912, 9216, 1056, 1056, 1056, 1056, 12000, 14112, 16224, 17280, 7968, 7968, 10272, 0, 0, 0, 0, 13056, 15168, 18336, 19392, 6912, 6912, 9216, 2304, 4608, 6720, 8832, 10944, 13056, 15168, 17280, 19392, 21504, 23616, 0, 15168, 19392, 6912, 9216, 11520, 31680, 28992, 29376, 29760, 26880, 20736, 0, 2304, 4608, 6912, 9216, 11520, 9216, 11520, 13824, 13824, 0, 0, 11520, 13824, 17280, 0, 2304, 2304, 30144, 27264, 21120, 21888, 3456, 6912, 25728, 16128, 17280, 13824, 18432, 4608, 0, 0, 0, 0, 2304, 5760, 1152, 1152, 1152, 1152, 4224, 4608, 23040, 3840, 7296, 8064, 12096, 24192, 14976, 19584, 20736, 21888, 23040, 18432, 0, 2304, 4608, 6912, 9216, 11520, 1152, 3456, 5760, 8448, 10368, 12672, 2304, 5760, 6912, 9216, 11520, 18432, 0, 2304, 4608, 6912, 9216, 11520, 9216, 11520, 18432, 20736, 0, 0, 11520, 19584, 20736, 0, 2304, 2304, 13824, 16128, 18432, 20736, 23040, 25344
};
alignas(16) const uint8_t offline_tensor_data[769104] = {
};

int CtxSummary(){
    printf("Arena Size: %d\n", arena_size);
    printf("Tensor Metadata Summary:\n");
    printf("\ttensors: %d\n",sizeof(tensors));
    printf("\tquant_scale: %d\n",27520);
    printf("\tquant_zeropoint: %d\n",1079);
    printf("\tsplit_offset: %d\n",sizeof(split_offset));
    printf("\toffline_tensor_data: %d\n",sizeof(offline_tensor_data));

    int byte_tensor =   sizeof(tensors) +
                        27520 + 1079 +
                        sizeof(split_offset) + sizeof(offline_tensor_data);
    return byte_tensor;
}