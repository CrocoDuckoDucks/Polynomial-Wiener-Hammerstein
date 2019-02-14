fi = library("filters.lib");
de = library("delays.lib");

br1 = fi.fir((-0.0, 1.907557864660443e-6, -1.3094405021124041e-5, 4.2415403348676764e-5, -9.95711796161365e-5, 0.00019506912771248192, -0.000340190512528335, 0.000546969872152571, -0.0008281945548262502, 0.0011974344262806796, -0.001669114815451523, 0.0022586500548103434, -0.0029826611785935533, 0.0038593105265088434, -0.004908799886849467, 0.006154100276494967, -0.007622015411532365, 0.009344736015002094, -0.011362134023039087, 0.013725203973914335, -0.016501341083991608, 0.019782669848947436, -0.023699658339966494, 0.028444356416384393, -0.034312223158252754, 0.04178254306705896, -0.05168652137653983, 0.06559942180791666, -0.08690480011217415, 0.12441266175602361, -0.21044979867378016, 0.636032668581742, 0.636032668581742, -0.21044979867378016, 0.12441266175602363, -0.08690480011217415, 0.06559942180791666, -0.05168652137653983, 0.04178254306705897, -0.03431222315825277, 0.0284443564163844, -0.023699658339966505, 0.019782669848947426, -0.01650134108399161, 0.013725203973914342, -0.01136213402303909, 0.0093447360150021, -0.0076220154115323666, 0.006154100276494967, -0.00490879988684947, 0.003859310526508844, -0.0029826611785935555, 0.0022586500548103443, -0.001669114815451522, 0.0011974344262806803, -0.0008281945548262499, 0.0005469698721525715, -0.0003401905125283359, 0.00019506912771248227, -9.95711796161365e-5, 4.2415403348676866e-5, -1.3094405021124041e-5, 1.9075578646604856e-6, -0.0)) : pow(_, 1) : *(_, 3.874555509347341) : *(_, 0.32872314429623967) : fi.tf2np(1.0, 1.0, 0.0, -0.3425537114075206, 0.0);
br2 = fi.fir((-0.0, -9.544266754281813e-7, 0.00010676990901891604, 2.1286214635612336e-5, -0.0004516843174511138, -9.843012172706948e-5, 0.0010707379087730867, 0.0002781850257732888, -0.0019993962815744223, -0.0006153697821585577, 0.0032745026999160837, 0.001175854160332379, -0.0049380887036659165, -0.0020406539295775333, 0.0070444057070777515, 0.003313983029032342, -0.009672814040415518, -0.005139174538332127, 0.012952263326317678, 0.007731301181507253, -0.017111067910239617, -0.011448676097971826, 0.022588781471658635, 0.01696704136451338, -0.03032555249542523, -0.02577717080755891, 0.04268012679738135, 0.042011248340639414, -0.06749039635966847, -0.08303747773090742, 0.15460601876999705, 0.44432440163677495, 0.44432440163677495, 0.15460601876999705, -0.08303747773090743, -0.06749039635966847, 0.042011248340639414, 0.04268012679738135, -0.025777170807558914, -0.030325552495425238, 0.016967041364513383, 0.02258878147165865, -0.011448676097971821, -0.01711106791023962, 0.007731301181507257, 0.01295226332631768, -0.00513917453833213, -0.00967281404041552, 0.003313983029032342, 0.007044405707077756, -0.0020406539295775338, -0.00493808870366592, 0.0011758541603323794, 0.003274502699916082, -0.0006153697821585581, -0.0019993962815744214, 0.000278185025773289, 0.0010707379087730895, -9.843012172706966e-5, -0.0004516843174511138, 2.128621463561239e-5, 0.00010676990901891604, -9.544266754282025e-7, -0.0)) : pow(_, 2) : *(_, 4.957182284650352) : *(_, 0.7059770331481197) : fi.tf2np(1.0, -1.8481911788290355, 1.0, -1.3047805251202484, 0.4119540662962394);
br3 = fi.fir((0.0, 6.363301735915225e-7, -9.037016543076986e-5, -0.00022159603424243724, -3.3429866262753906e-5, 0.0005979360359496387, 0.0009914161056646018, 0.00018603043495267444, -0.0016110644687395003, -0.0024846770296260683, -0.0005789025380980347, 0.003196612088575052, 0.00491825566639206, 0.0013743365363159518, -0.005442658967005377, -0.008601180418038831, -0.0028067518469597165, 0.00850919204461152, 0.014061680196201933, 0.005265654866936345, -0.012756839151326567, -0.022400515636554565, -0.009548487238537776, 0.019141062769200183, 0.03649801841420166, 0.017831089736625852, -0.030878497903997947, -0.0670473444446577, -0.03965514507846828, 0.06706130974857814, 0.21094692066774037, 0.3135773091458267, 0.3135773091458267, 0.21094692066774037, 0.06706130974857814, -0.03965514507846828, -0.0670473444446577, -0.030878497903997947, 0.017831089736625855, 0.03649801841420167, 0.019141062769200187, -0.009548487238537781, -0.022400515636554555, -0.012756839151326568, 0.005265654866936347, 0.014061680196201937, 0.008509192044611525, -0.0028067518469597173, -0.008601180418038831, -0.00544265896700538, 0.001374336536315952, 0.004918255666392064, 0.0031966120885750533, -0.0005789025380980344, -0.0024846770296260696, -0.0016110644687394996, 0.00018603043495267463, 0.0009914161056646042, 0.0005979360359496398, -3.3429866262753906e-5, -0.0002215960342424378, -9.037016543076986e-5, 6.363301735915367e-7, 0.0)) : pow(_, 3) : *(_, 4.236820036674279) : *(_, 0.925469217445072) : fi.tf2np(1.0, 1.997765930831419, 0.9999999999999996, 1.977987133923754, 0.9807677092872449);
br4 = fi.fir((-0.0, -2.592248301723185e-5, -7.791407656335104e-5, -1.0651476875052093e-5, 0.0003030789152101295, 0.0007296185880668887, 0.0008254185478474729, 0.00013966465340451993, -0.0012866217115476326, -0.0026448287295592772, -0.002681968857284718, -0.0005937496953480618, 0.003059694527653959, 0.006186212292613543, 0.006170007826022992, 0.0016871956263728235, -0.005792264144318499, -0.01205589774788589, -0.01222197551517411, -0.003978494087520307, 0.009936919121867175, 0.021946645370531134, 0.023162432667999833, 0.008847742739284823, -0.017134668295575594, -0.041860775385647024, -0.048043422951271245, -0.02225826632526469, 0.03721679677470309, 0.1172144046980106, 0.19341283308691964, 0.2398287560463441, 0.2398287560463441, 0.19341283308691964, 0.11721440469801062, 0.03721679677470309, -0.02225826632526469, -0.048043422951271245, -0.04186077538564703, -0.0171346682955756, 0.008847742739284825, 0.023162432667999847, 0.021946645370531124, 0.009936919121867177, -0.0039784940875203095, -0.012221975515174111, -0.012055897747885898, -0.0057922641443185, 0.0016871956263728235, 0.006170007826022996, 0.006186212292613544, 0.0030596945276539613, -0.000593749695348062, -0.0026819688572847166, -0.0026448287295592786, -0.0012866217115476322, 0.00013966465340452006, 0.0008254185478474749, 0.0007296185880668901, 0.0003030789152101295, -1.065147687505212e-5, -7.791407656335104e-5, -2.592248301723243e-5, -0.0)) : pow(_, 4) : *(_, 36.1537453893171) : *(_, 0.3034775529523943) : fi.tf2np(1.0, -1.710685626661486, 1.0, -0.5191546878500608, -0.39304489409521126);
br5 = fi.fir((0.0, 3.816978989119905e-7, -6.071023262924333e-5, -0.00023309422100835337, -0.0004378246492934508, -0.0004610286119422451, -6.913320176345892e-5, 0.0008001431372561246, 0.0018835627030676359, 0.002597568456150338, 0.0022609170116879196, 0.000475452421901315, -0.002507869683718091, -0.00565529079824068, -0.007401863830994233, -0.006277441598499712, -0.0016966847881475778, 0.005438116269332021, 0.012672480826595745, 0.016654929813464354, 0.014401935877149812, 0.004761745525651816, -0.010509509910449853, -0.02659809368277125, -0.0366629565665558, -0.03386486110618535, -0.013790242266349275, 0.023570908539581464, 0.07306926521835808, 0.1252701090702349, 0.16881097166006426, 0.19355811692015382, 0.19355811692015382, 0.16881097166006426, 0.12527010907023492, 0.07306926521835808, 0.023570908539581464, -0.013790242266349275, -0.03386486110618535, -0.03666295656655581, -0.026598093682771256, -0.010509509910449858, 0.004761745525651814, 0.014401935877149816, 0.01665492981346436, 0.012672480826595748, 0.005438116269332024, -0.0016966847881475782, -0.006277441598499712, -0.007401863830994237, -0.0056552907982406805, -0.002507869683718093, 0.00047545242190131517, 0.0022609170116879187, 0.0025975684561503395, 0.0018835627030676352, 0.0008001431372561253, -6.913320176345909e-5, -0.0004610286119422459, -0.0004378246492934508, -0.00023309422100835394, -6.071023262924333e-5, 3.8169789891199903e-7, 0.0)) : pow(_, 5) : *(_, 3.8583159825100943) : *(_, 0.03725032448878116) : fi.tf2np(1.0, 1.0, 0.0, -0.9254993510224376, 0.0);
br6 = fi.fir((-0.0, -3.1842789782240263e-7, 5.16157344895289e-5, 0.0002113674601069362, 0.00045446189731789243, 0.0006496575301148225, 0.0005918224051902988, 9.32605461530386e-5, -0.0008970288424174376, -0.0021976333408804845, -0.0033761960694798725, -0.0038409997341920113, -0.00303652532033853, -0.0006918998649275151, 0.002963041274254465, 0.007107745686385434, 0.010435592129486491, 0.011466718614747613, 0.009012145975850604, 0.002668692402141174, -0.006809924181899454, -0.017409313871300767, -0.02610265723594003, -0.029441784770069635, -0.024378348392384386, -0.009119982618641758, 0.016207637352193875, 0.049292959061984436, 0.08587647697876556, 0.12047918439636057, 0.14747532364825952, 0.16226490957656745, 0.16226490957656745, 0.14747532364825952, 0.12047918439636059, 0.08587647697876556, 0.049292959061984436, 0.016207637352193875, -0.009119982618641758, -0.024378348392384393, -0.029441784770069642, -0.026102657235940047, -0.01740931387130076, -0.006809924181899455, 0.0026686924021411752, 0.009012145975850606, 0.01146671861474762, 0.010435592129486495, 0.007107745686385434, 0.0029630412742544664, -0.0006918998649275152, -0.003036525320338532, -0.003840999734192013, -0.003376196069479871, -0.0021976333408804858, -0.0008970288424174373, 9.32605461530387e-5, 0.0005918224051903002, 0.0006496575301148237, 0.00045446189731789243, 0.00021136746010693672, 5.16157344895289e-5, -3.184278978224097e-7, -0.0)) : pow(_, 6) : *(_, 1.9087747472069876) : *(_, 0.8677493986663203) : fi.tf2np(1.0, -1.0, 0.0, -0.7354987973326406, 0.0);
br7 = fi.fir((0.0, 2.043198855930036e-5, 4.807155798342314e-5, 6.085623334044402e-6, -0.00018407169277486954, -0.0005535347706195292, -0.0010447047526428558, -0.0014966858446036869, -0.0016700748905827854, -0.001312067547280547, -0.00024899012577765846, 0.0015186490771680614, 0.0037497068167890715, 0.005953377614065725, 0.007455096050662157, 0.007536516891052996, 0.005627550393271911, 0.0015099241325991974, -0.004518731301358437, -0.011569917816753431, -0.01821380013998131, -0.022663572301803044, -0.023074309066188396, -0.017906576982010897, -0.006284262940760564, 0.011730466277649038, 0.035011745408095395, 0.06143460704249418, 0.08813304814142653, 0.11191492975820208, 0.12976272395126534, 0.13932836944851956, 0.13932836944851956, 0.12976272395126534, 0.1119149297582021, 0.08813304814142653, 0.06143460704249418, 0.035011745408095395, 0.01173046627764904, -0.0062842629407605655, -0.0179065769820109, -0.02307430906618841, -0.022663572301803037, -0.01821380013998131, -0.011569917816753438, -0.004518731301358437, 0.0015099241325991983, 0.005627550393271912, 0.007536516891052996, 0.0074550960506621616, 0.0059533776140657255, 0.003749706816789074, 0.0015186490771680618, -0.00024899012577765835, -0.0013120675472805478, -0.0016700748905827847, -0.0014966858446036884, -0.0010447047526428587, -0.0005535347706195302, -0.00018407169277486954, 6.0856233340444175e-6, 4.807155798342314e-5, 2.0431988559300816e-5, 0.0)) : pow(_, 7) : *(_, 114.62389984607816) : *(_, 0.809020197362308) : fi.tf2np(1.0, -2.0, 1.0, -1.5812283276382926, 0.65485246181094);
br8 = fi.fir((-0.0, -1.8139994019341248e-5, -9.804817324745739e-5, -0.00024768598309025426, -0.00042389277891318605, -0.0005335554764333956, -0.0004539928593901172, -6.981563869182546e-5, 0.0006816194317156873, 0.001767765437381579, 0.0030342813294551916, 0.004205254498554649, 0.004916421784324883, 0.004780575271646264, 0.0034765549765896357, 0.0008463280706153401, -0.003019727285311521, -0.007729841314808856, -0.012587020575199876, -0.01664359374141537, -0.0188129870140008, -0.01802630065495888, -0.01341125843788163, -0.004464463850101964, 0.008813827986493878, 0.025849344296919312, 0.04550566093667687, 0.06618320099869036, 0.08599570032892284, 0.10300086526335078, 0.11545180740392787, 0.12203111576219919, 0.12203111576219919, 0.11545180740392787, 0.1030008652633508, 0.08599570032892284, 0.06618320099869036, 0.04550566093667687, 0.025849344296919315, 0.00881382798649388, -0.0044644638501019655, -0.01341125843788164, -0.018026300654958874, -0.018812987014000804, -0.01664359374141538, -0.012587020575199878, -0.007729841314808861, -0.003019727285311522, 0.0008463280706153401, 0.0034765549765896375, 0.004780575271646264, 0.004916421784324885, 0.004205254498554651, 0.0030342813294551903, 0.00176776543738158, 0.0006816194317156871, -6.981563869182552e-5, -0.0004539928593901183, -0.0005335554764333964, -0.00042389277891318605, -0.0002476859830902549, -9.804817324745739e-5, -1.8139994019341655e-5, -0.0)) : pow(_, 8) : *(_, 1.2148845229049379) : *(_, 0.9661412616011574) : fi.tf2np(1.0, -1.0, 0.0, -0.932282523202315, 0.0);
br9 = fi.fir((-0.0, -2.2586274732998145e-5, -6.994673856543064e-5, -8.938672472940644e-5, -1.1164981019235727e-5, 0.00022967137183825084, 0.00066795077646096, 0.001285410557316531, 0.0019962217450744625, 0.002646116292805432, 0.0030280454526359587, 0.0029142699574603373, 0.0021014190781655597, 0.0004619687864390614, -0.002006614151615106, -0.00514464571819124, -0.008613680458963059, -0.011906945562377256, -0.014389222895390576, -0.015364014774476232, -0.014161114292837071, -0.010233692121945566, -0.0032514248577451338, 0.006824454919729711, 0.019696782491541987, 0.034727777321034575, 0.0509778950403487, 0.06728601537140469, 0.08238275319565405, 0.09502362058634785, 0.10412546403675538, 0.10888860257157477, 0.10888860257157477, 0.10412546403675538, 0.09502362058634786, 0.08238275319565405, 0.06728601537140469, 0.0509778950403487, 0.034727777321034575, 0.01969678249154199, 0.006824454919729713, -0.0032514248577451355, -0.01023369212194556, -0.014161114292837075, -0.01536401477447624, -0.014389222895390578, -0.011906945562377263, -0.00861368045896306, -0.00514464571819124, -0.0020066141516151075, 0.00046196878643906143, 0.002101419078165561, 0.0029142699574603386, 0.003028045452635957, 0.0026461162928054336, 0.001996221745074462, 0.0012854105573165322, 0.0006679507764609618, 0.00022967137183825127, -1.1164981019235727e-5, -8.938672472940667e-5, -6.994673856543064e-5, -2.2586274732998646e-5, -0.0)) : pow(_, 9) : *(_, 2.011842968793935) : *(_, 0.16332305357115232) : fi.tf2np(1.0, 1.0, 0.0, -0.6733538928576954, 0.0);
br10 = fi.fir((-0.0, -1.9160102114588302e-7, 3.191730529284115e-5, 0.00014286183284853692, 0.0003629480093577033, 0.0006916666691054791, 0.0010877857959688377, 0.001466890118111902, 0.001708074856706803, 0.0016700458332489258, 0.0012152424692788444, 0.00023903730459107783, -0.001300166701432065, -0.00335479990846251, -0.005770787978903894, -0.008283075526107185, -0.010526610097535784, -0.012063398613646768, -0.01242399703644272, -0.011159588676828867, -0.00789896834250778, -0.002403563507993082, 0.005386704176653444, 0.015323289392251273, 0.027039657111544435, 0.03996541880051379, 0.05336394039395777, 0.06639049600301349, 0.07816539813575238, 0.08785449987470034, 0.09474831961328527, 0.09833095429469874, 0.09833095429469874, 0.09474831961328527, 0.08785449987470036, 0.07816539813575238, 0.06639049600301349, 0.05336394039395777, 0.03996541880051379, 0.027039657111544442, 0.015323289392251277, 0.005386704176653447, -0.0024035635079930812, -0.007898968342507782, -0.011159588676828872, -0.012423997036442721, -0.012063398613646775, -0.010526610097535785, -0.008283075526107185, -0.0057707879789038975, -0.0033547999084625113, -0.001300166701432066, 0.00023903730459107791, 0.0012152424692788437, 0.0016700458332489269, 0.0017080748567068023, 0.0014668901181119035, 0.0010877857959688405, 0.0006916666691054804, 0.0003629480093577033, 0.00014286183284853727, 3.191730529284115e-5, -1.916010211458873e-7, -0.0)) : pow(_, 10) : *(_, 101.7862890313022) : *(_, 0.5080102498067397) : fi.tf2np(1.0, -1.9957560133122456, 1.0000000000000002, -0.9836443352889076, 0.4865408187229249) : fi.tf2np(1.0, -1.9957560133122456, 1.0000000000000002, -1.9956495664309113, 0.995671682996886);

process = _ <: br1, br2, br3, br4, br5, br6, br7, br8, br9, br10 :> _;
