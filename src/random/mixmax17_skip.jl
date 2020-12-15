const polyCoeffs17 = [2305843009213693950, 2301339409586323472, 2305812222886150007, 2304950122020066599, 432341041511554329, 218546466595982445, 2288774021294666627, 1327872805097898761, 1780609199527208287, 1649816982904334450, 870691130730945933, 845235595464045280, 1023515427770432269, 2127882104260287640, 2304992536952056903, 2305796279969513047, 2305843009213693918, 1]


#ei coefficients for m=2^5...2^30
const skipMat17 = [
    [
        1972101396566503791,
        2051641146438145039,
        533173487516893382,
        545730390508709012,
        1620905962071234677,
        1870596525612362239,
        2112925984824714998,
        868498104066040073,
        1148789919911044037,
        804191351341168970,
        730557211712521605,
        1517348237025551691,
        850529375445951650,
        717602406832799554,
        2195246085431969997,
        366890617173684729,
        1334285358249915285,
    ],
    [
        327007261501768800,
        1668507619896500323,
        570663528499880382,
        849832518716040141,
        167129205132215511,
        2029511566136161224,
        1622479814610858826,
        136068578967795545,
        465570142067313353,
        443857812536418275,
        1789415231111220632,
        722738813040608858,
        1536033290496530295,
        1803331387035840455,
        1337554956082899184,
        1262248399564869336,
        1964812692109901433,
    ],
    [
        1383941110888204807,
        1660111263786065031,
        635055952618544659,
        1275158498626512324,
        897543831095138544,
        1662589174551273379,
        1474986199793476754,
        683298560121080184,
        1364215338737587469,
        2093515129575752146,
        1526800205169804711,
        2260714998735099815,
        2305114977222035959,
        2007484595445621488,
        1660849571985359945,
        413454328436564294,
        1737301931246044859,
    ],
    [
        34041530522269951,
        989904588027666109,
        1424411770440999775,
        2233548010652125537,
        478899446812575694,
        1782594547279649086,
        1366273273270145235,
        771683963354479181,
        1042197874804458977,
        47908093511613389,
        41949539543464032,
        2092607021509534694,
        1553302290867055291,
        736326122801287713,
        1087073336441938432,
        1684732520189930480,
        1523208273561944990,
    ],
    [
        66724810607261494,
        2243218526996946637,
        1995426020901038827,
        1273288642662700447,
        1060296628596275187,
        1033117315917689018,
        1301796675675435740,
        1169744488162372532,
        1921449921433723255,
        1937495610468571249,
        993465857439554119,
        16019142551863611,
        2219970175835711455,
        1006443666620132302,
        924296262882437568,
        2208413115601089136,
        264904656702381098,
    ],
    [
        86455199185446885,
        747612738103107959,
        1407924622912058943,
        198804599081851031,
        25930318115067972,
        412979531876469963,
        715204133791195588,
        1086783985939604076,
        804486103308535628,
        137063081601334719,
        822767962759173950,
        1264697699255051205,
        1772814692491721476,
        2011485527677807227,
        399961272695350873,
        373953638541612193,
        11760493978029076,
    ],
    [
        1708255563439812290,
        774500572940894871,
        99516351798717781,
        231082297833576357,
        2060166097000388081,
        1966934547960002972,
        1891076714188958609,
        37390213127216786,
        1584006554430522049,
        890578844293885408,
        1954994145879248940,
        2062392734010795911,
        1020363785670620886,
        2100645351290474524,
        1589146708828507128,
        1348093719303308944,
        180639173108775193,
    ],
    [
        2202229385283005142,
        1571862202556900423,
        368225414353564040,
        119278548399139866,
        952453769547029233,
        1720919561984421856,
        1430751071426459339,
        1307940815395225146,
        1377228865567960952,
        1990790714390702490,
        677872157032915967,
        390217894376331041,
        738924010463110026,
        224005659670894829,
        2067169282375116419,
        1185837644893947828,
        628166474873784103,
    ],
    [
        909535552350355334,
        1984196621966487204,
        1203870959107599346,
        1777238049233603922,
        717242522487790393,
        345373014150664306,
        285368686010812695,
        1609391685732340886,
        739922468389050788,
        234977750553506588,
        1990140987404995049,
        1604923589910519614,
        2278523533111581954,
        180283447256677998,
        1068630207724457873,
        304098965825306400,
        2062051512419880902,
    ],
    [
        1422118065708901004,
        836642887813982710,
        1888061742065740657,
        274692856042566127,
        1356911043266965940,
        976761670652707464,
        531998614533163061,
        1912546600455521925,
        17045467741578191,
        1654907078127935108,
        2030183713003610570,
        352607571297516120,
        355680148820201396,
        739542852109405450,
        709616794056412549,
        1336256981892236022,
        2069426500410221871,
    ],
    [
        2286984874264166784,
        2166094372062800745,
        879293621066552442,
        2278089229880928468,
        1994187841709719990,
        2093302072134305223,
        361239617152043736,
        2130415391463653737,
        1938772631940986057,
        805733948963353760,
        1424018902747763999,
        773476430100927128,
        2124442120407728160,
        121106965413699034,
        475872283540920661,
        1044131603014983440,
        1923430800318934442,
    ],
    [
        451619876923000687,
        71097883912081557,
        1628549684595849797,
        41885163308977346,
        362429429797508426,
        971288200022972701,
        1418629218562813163,
        752202962754801305,
        2086557484323717914,
        626769727155832314,
        1899885130221156377,
        298197818644716470,
        245633358844767262,
        976528303012507716,
        1559467768505760317,
        44483913443667677,
        1725590647900820418,
    ],
    [
        661193250562452382,
        1043865391393629623,
        607105552343853366,
        1082113160102069187,
        1808845276619195583,
        2140932414351717404,
        852505500160497633,
        1975416625595591040,
        1230888852709529781,
        523574463061188471,
        176632315048861766,
        735704925835568324,
        821608995808733894,
        2236293070564040013,
        1772032514358201802,
        1816015347932134306,
        1317895217324711671,
    ],
    [
        620079138192442768,
        1413852673829294104,
        139241503827318837,
        890594028110926166,
        977736786367659953,
        742757423130029561,
        23963199723774009,
        951576424938956514,
        237070967500044720,
        709309519512381854,
        1217415377215609678,
        2010236133988708113,
        1922119966574907981,
        2116003072079736225,
        1058675710956122435,
        2291324972928830992,
        1412559516146851951,
    ],
    [
        1404725383601173936,
        1109185334627610252,
        2177895433232027303,
        1454696689462755000,
        559089231530930806,
        2113042216679845782,
        2212705677772153535,
        58693736675718155,
        745821794534861716,
        2071125041882463283,
        1291815773765129120,
        1446497568380515246,
        328849553205520604,
        1572459871580105595,
        770521934324544238,
        2162594601603215507,
        1790149936455713032,
    ],
    [
        843356255388265439,
        2303128525550447885,
        1403805614380726589,
        1646775209363665369,
        1656489684429545237,
        136962756483849705,
        578486145972684395,
        1186788852536058806,
        744606065061259291,
        129235498787499648,
        987271125576440333,
        2226012411473073226,
        1064839699140268466,
        1194475380559801816,
        1591779369875059359,
        1167968540715842034,
        2137709594798027207,
    ],
    [
        1972101396566503791,
        2051641146438145039,
        533173487516893382,
        545730390508709012,
        1620905962071234677,
        1870596525612362239,
        2112925984824714998,
        868498104066040073,
        1148789919911044037,
        804191351341168970,
        730557211712521605,
        1517348237025551691,
        850529375445951650,
        717602406832799554,
        2195246085431969997,
        366890617173684729,
        1334285358249915285,
    ],
    [
        327007261501768800,
        1668507619896500323,
        570663528499880382,
        849832518716040141,
        167129205132215511,
        2029511566136161224,
        1622479814610858826,
        136068578967795545,
        465570142067313353,
        443857812536418275,
        1789415231111220632,
        722738813040608858,
        1536033290496530295,
        1803331387035840455,
        1337554956082899184,
        1262248399564869336,
        1964812692109901433,
    ],
    [
        1383941110888204807,
        1660111263786065031,
        635055952618544659,
        1275158498626512324,
        897543831095138544,
        1662589174551273379,
        1474986199793476754,
        683298560121080184,
        1364215338737587469,
        2093515129575752146,
        1526800205169804711,
        2260714998735099815,
        2305114977222035959,
        2007484595445621488,
        1660849571985359945,
        413454328436564294,
        1737301931246044859,
    ],
    [
        34041530522269951,
        989904588027666109,
        1424411770440999775,
        2233548010652125537,
        478899446812575694,
        1782594547279649086,
        1366273273270145235,
        771683963354479181,
        1042197874804458977,
        47908093511613389,
        41949539543464032,
        2092607021509534694,
        1553302290867055291,
        736326122801287713,
        1087073336441938432,
        1684732520189930480,
        1523208273561944990,
    ],
    [
        66724810607261494,
        2243218526996946637,
        1995426020901038827,
        1273288642662700447,
        1060296628596275187,
        1033117315917689018,
        1301796675675435740,
        1169744488162372532,
        1921449921433723255,
        1937495610468571249,
        993465857439554119,
        16019142551863611,
        2219970175835711455,
        1006443666620132302,
        924296262882437568,
        2208413115601089136,
        264904656702381098,
    ],
    [
        86455199185446885,
        747612738103107959,
        1407924622912058943,
        198804599081851031,
        25930318115067972,
        412979531876469963,
        715204133791195588,
        1086783985939604076,
        804486103308535628,
        137063081601334719,
        822767962759173950,
        1264697699255051205,
        1772814692491721476,
        2011485527677807227,
        399961272695350873,
        373953638541612193,
        11760493978029076,
    ],
    [
        1708255563439812290,
        774500572940894871,
        99516351798717781,
        231082297833576357,
        2060166097000388081,
        1966934547960002972,
        1891076714188958609,
        37390213127216786,
        1584006554430522049,
        890578844293885408,
        1954994145879248940,
        2062392734010795911,
        1020363785670620886,
        2100645351290474524,
        1589146708828507128,
        1348093719303308944,
        180639173108775193,
    ],
    [
        2202229385283005142,
        1571862202556900423,
        368225414353564040,
        119278548399139866,
        952453769547029233,
        1720919561984421856,
        1430751071426459339,
        1307940815395225146,
        1377228865567960952,
        1990790714390702490,
        677872157032915967,
        390217894376331041,
        738924010463110026,
        224005659670894829,
        2067169282375116419,
        1185837644893947828,
        628166474873784103,
    ],
    [
        909535552350355334,
        1984196621966487204,
        1203870959107599346,
        1777238049233603922,
        717242522487790393,
        345373014150664306,
        285368686010812695,
        1609391685732340886,
        739922468389050788,
        234977750553506588,
        1990140987404995049,
        1604923589910519614,
        2278523533111581954,
        180283447256677998,
        1068630207724457873,
        304098965825306400,
        2062051512419880902,
    ],
    [
        1422118065708901004,
        836642887813982710,
        1888061742065740657,
        274692856042566127,
        1356911043266965940,
        976761670652707464,
        531998614533163061,
        1912546600455521925,
        17045467741578191,
        1654907078127935108,
        2030183713003610570,
        352607571297516120,
        355680148820201396,
        739542852109405450,
        709616794056412549,
        1336256981892236022,
        2069426500410221871,
    ],
    [
        2286984874264166784,
        2166094372062800745,
        879293621066552442,
        2278089229880928468,
        1994187841709719990,
        2093302072134305223,
        361239617152043736,
        2130415391463653737,
        1938772631940986057,
        805733948963353760,
        1424018902747763999,
        773476430100927128,
        2124442120407728160,
        121106965413699034,
        475872283540920661,
        1044131603014983440,
        1923430800318934442,
    ],
    [
        451619876923000687,
        71097883912081557,
        1628549684595849797,
        41885163308977346,
        362429429797508426,
        971288200022972701,
        1418629218562813163,
        752202962754801305,
        2086557484323717914,
        626769727155832314,
        1899885130221156377,
        298197818644716470,
        245633358844767262,
        976528303012507716,
        1559467768505760317,
        44483913443667677,
        1725590647900820418,
    ],
    [
        661193250562452382,
        1043865391393629623,
        607105552343853366,
        1082113160102069187,
        1808845276619195583,
        2140932414351717404,
        852505500160497633,
        1975416625595591040,
        1230888852709529781,
        523574463061188471,
        176632315048861766,
        735704925835568324,
        821608995808733894,
        2236293070564040013,
        1772032514358201802,
        1816015347932134306,
        1317895217324711671,
    ],
    [
        620079138192442768,
        1413852673829294104,
        139241503827318837,
        890594028110926166,
        977736786367659953,
        742757423130029561,
        23963199723774009,
        951576424938956514,
        237070967500044720,
        709309519512381854,
        1217415377215609678,
        2010236133988708113,
        1922119966574907981,
        2116003072079736225,
        1058675710956122435,
        2291324972928830992,
        1412559516146851951,
    ],
    [
        1404725383601173936,
        1109185334627610252,
        2177895433232027303,
        1454696689462755000,
        559089231530930806,
        2113042216679845782,
        2212705677772153535,
        58693736675718155,
        745821794534861716,
        2071125041882463283,
        1291815773765129120,
        1446497568380515246,
        328849553205520604,
        1572459871580105595,
        770521934324544238,
        2162594601603215507,
        1790149936455713032,
    ],
    [
        843356255388265439,
        2303128525550447885,
        1403805614380726589,
        1646775209363665369,
        1656489684429545237,
        136962756483849705,
        578486145972684395,
        1186788852536058806,
        744606065061259291,
        129235498787499648,
        987271125576440333,
        2226012411473073226,
        1064839699140268466,
        1194475380559801816,
        1591779369875059359,
        1167968540715842034,
        2137709594798027207,
    ],
    [
        1603310798512369292,
        1601100715499888665,
        1072379507175706577,
        1213079833732518096,
        2001446403012491729,
        419527915496992901,
        1536979475280290440,
        301233389316038205,
        1272639558366606573,
        1557584358952356730,
        220291452971595398,
        2054031811190278086,
        2102439523114162782,
        2043691020304126326,
        931868944669449976,
        1695482396558042263,
        569471942240552669,
    ],
    [
        694964758088788671,
        1038219072747505180,
        1918123200517156133,
        2205661282197833706,
        273525348357470113,
        2121494391145814980,
        284345776977803033,
        246418780931512795,
        2283919448816535183,
        1825147280778299237,
        1690836788966579901,
        668223754085848325,
        1157160865829805369,
        948478357813085584,
        254359412607383270,
        1135224685685690391,
        2000460937859975771,
    ],
    [
        707900843551449518,
        824721369516455951,
        1253414446894639545,
        2054381172664740386,
        2262749485189772637,
        209547095357435226,
        1486285367660805487,
        1066212727062337001,
        138467865375837985,
        502055498606674686,
        1631322852248214521,
        2099497052471449902,
        653339673532849910,
        1209350158362883423,
        801199054076992563,
        114429634116085316,
        1756481675028861979,
    ],
    [
        1233556889083976385,
        2271316709310852100,
        444255811069641330,
        1920402384108973078,
        1871076587612059977,
        848766558121280411,
        183916672241888601,
        861892142014135059,
        1662987476773654122,
        886432897949545677,
        952076682704709636,
        2191551637233270217,
        1294187932820418508,
        691043740128684730,
        1580114045565902135,
        871127380587503701,
        1982811410939956429,
    ],
    [
        1026536596545411590,
        672006064876756356,
        187865928299266937,
        1333180393912329299,
        449284825643474283,
        1880633394205436049,
        2130751968571655233,
        1666493909078411462,
        1226379145018641537,
        925073470477791740,
        1893981127322123262,
        1681869522295985310,
        563899279485833974,
        1238859114847806213,
        1910620991062119273,
        573004330562037184,
        2305020818618415056,
    ],
    [
        1649536426891372496,
        1773496319691523181,
        1901485078913394927,
        723526059786718651,
        1070993657770771508,
        561598982654578851,
        514075350227526556,
        1093001656896099183,
        1131031199773443838,
        1567007165576640890,
        1942807866844241746,
        1377184921804775466,
        525984842795669092,
        1310907835907728052,
        1744624221468767441,
        1158478818307902549,
        178013669142197363,
    ],
    [
        348651919527231585,
        1956322108668706848,
        2200609920440762066,
        1722438664965020838,
        2287184028526454944,
        1503209766683435682,
        644324372269330068,
        839244854502053928,
        822962662761844987,
        967028104753664904,
        2219260946362335472,
        643036698992345239,
        767336458352242411,
        1856163925694015939,
        310679063625611629,
        1008184523753394922,
        1557936276546470810,
    ],
    [
        1358639256207203914,
        380878389447600344,
        266118891978454911,
        1913066921087211778,
        1415556213238755288,
        1041859734009513367,
        913429674643150097,
        1281916063862733602,
        649226289994219281,
        2012458594853276202,
        806265659335377279,
        488631900297180252,
        1174216778495607541,
        66231138464254448,
        102026227548073028,
        6202600785403592,
        1876707885948960382,
    ],
    [
        302708009212573930,
        640510323803542134,
        390948779667971340,
        300092081045236432,
        1387655392749610816,
        973283902592015763,
        1559965596923378008,
        1181369898028308254,
        914040186129074102,
        900388984624711127,
        409569314904400659,
        423416875652327473,
        1646832545602879919,
        1122149242942110197,
        1318635598796713194,
        27210210178262476,
        851295614470433169,
    ],
    [
        1991387975529265218,
        1843351940963099488,
        2271846316599365716,
        2132592891396609803,
        2302343546822788973,
        2129811222706328476,
        1254066806718120786,
        2268141813723625983,
        1450651840911637570,
        726351353821067277,
        2126412864417917107,
        952525217820829278,
        1116731824858793647,
        2018141859931748691,
        1074656985216627374,
        536391227666488033,
        346611616870907958,
    ],
]