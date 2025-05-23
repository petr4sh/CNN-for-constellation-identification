from PIL import Image
import pickle
import numpy as np


def star(file):
    img = Image.open(file)
    pixels = img.load()
    w, h = img.size
    braight = []
    for x in range(w):
        for y in range(h):
            r, g, b = pixels[x, y]
            braight.append((r + g + b) // 3)
    br = np.array(braight)
    return br


elements, s = [],  0
for i in range(1, 86):
    while s < 12:
        elements.append(i)
        s += 1
    s = 0


data = {'Train': {'Labels': np.array(elements),
                  'Features': np.array([star('андромеда.jpg'), star('андромеда1.jpg'), star('андромеда2.jpg'), star('андромеда3.jpg'),
                                        star('андромеда4.jpg'), star('андромеда5.jpg'), star('андромеда6.jpg'), star('андромеда7.jpg'),
                                        star('андромеда8.jpg'), star('андромеда9.jpg'), star('андромеда10.jpg'), star('андромеда11.jpg'),
                                        star('близнецы.jpg'), star('близнецы1.jpg'), star('близнецы2.jpg'), star('близнецы3.jpg'),
                                        star('близнецы4.jpg'), star('близнецы5.jpg'), star('близнецы6.jpg'), star('близнецы7.jpg'),
                                        star('близнецы8.jpg'), star('близнецы9.jpg'), star('близнецы10.jpg'), star('близнецы11.jpg'),
                                        star('большая медведица.jpg'), star('большая медведица1.jpg'), star('большая медведица2.jpg'), star('большая медведица3.jpg'),
                                        star('большая медведица4.jpg'), star('большая медведица5.jpg'), star('большая медведица6.jpg'), star('большая медведица7.jpg'),
                                        star('большая медведица8.jpg'), star('большая медведица9.jpg'), star('большая медведица10.jpg'), star('большая медведица11.jpg'),
                                        star('большой пес.jpg'), star('большой пес1.jpg'), star('большой пес2.jpg'), star('большой пес3.jpg'),
                                        star('большой пес4.jpg'), star('большой пес5.jpg'), star('большой пес6.jpg'), star('большой пес7.jpg'),
                                        star('большой пес8.jpg'), star('большой пес9.jpg'), star('большой пес10.jpg'), star('большой пес11.jpg'),
                                        star('весы.jpg'), star('весы1.jpg'), star('весы2.jpg'), star('весы3.jpg'),
                                        star('весы4.jpg'), star('весы5.jpg'), star('весы6.jpg'), star('весы7.jpg'),
                                        star('весы8.jpg'), star('весы9.jpg'), star('весы10.jpg'), star('весы11.jpg'),
                                        star('водолей.jpg'), star('водолей1.jpg'), star('водолей2.jpg'), star('водолей3.jpg'),
                                        star('водолей4.jpg'), star('водолей5.jpg'), star('водолей6.jpg'), star('водолей7.jpg'),
                                        star('водолей8.jpg'), star('водолей9.jpg'), star('водолей10.jpg'), star('водолей11.jpg'),
                                        star('возничий.jpg'), star('возничий1.jpg'), star('возничий2.jpg'), star('возничий3.jpg'),
                                        star('возничий4.jpg'), star('возничий5.jpg'), star('возничий6.jpg'), star('возничий7.jpg'),
                                        star('возничий8.jpg'), star('возничий9.jpg'), star('возничий10.jpg'), star('возничий11.jpg'),
                                        star('волк.jpg'), star('волк1.jpg'), star('волк2.jpg'), star('волк3.jpg'),
                                        star('волк4.jpg'), star('волк5.jpg'), star('волк6.jpg'), star('волк7.jpg'),
                                        star('волк8.jpg'), star('волк9.jpg'), star('волк10.jpg'), star('волк11.jpg'),
                                        star('волопас.jpg'), star('волопас1.jpg'), star('волопас2.jpg'), star('волопас3.jpg'),
                                        star('волопас4.jpg'), star('волопас5.jpg'), star('волопас6.jpg'), star('волопас7.jpg'),
                                        star('волопас8.jpg'), star('волопас9.jpg'), star('волопас10.jpg'), star('волопас11.jpg'),
                                        star('волосы вероники.jpg'), star('волосы вероники1.jpg'), star('волосы вероники2.jpg'), star('волосы вероники3.jpg'),
                                        star('волосы вероники4.jpg'), star('волосы вероники5.jpg'), star('волосы вероники6.jpg'), star('волосы вероники7.jpg'),
                                        star('волосы вероники8.jpg'), star('волосы вероники9.jpg'), star('волосы вероники10.jpg'), star('волосы вероники11.jpg'),
                                        star('ворон.jpg'), star('ворон1.jpg'), star('ворон2.jpg'), star('ворон3.jpg'),
                                        star('ворон4.jpg'), star('ворон5.jpg'), star('ворон6.jpg'), star('ворон7.jpg'),
                                        star('ворон8.jpg'), star('ворон9.jpg'), star('ворон10.jpg'), star('ворон11.jpg'),
                                        star('геркулес.jpg'), star('геркулес1.jpg'), star('геркулес2.jpg'), star('геркулес3.jpg'),
                                        star('геркулес4.jpg'), star('геркулес5.jpg'), star('геркулес6.jpg'), star('геркулес7.jpg'),
                                        star('геркулес8.jpg'), star('геркулес9.jpg'), star('геркулес10.jpg'), star('геркулес11.jpg'),
                                        star('гидра.jpg'), star('гидра1.jpg'), star('гидра2.jpg'), star('гидра3.jpg'),
                                        star('гидра4.jpg'), star('гидра5.jpg'), star('гидра6.jpg'), star('гидра7.jpg'),
                                        star('гидра8.jpg'), star('гидра9.jpg'), star('гидра10.jpg'), star('гидра11.jpg'),
                                        star('голубь.jpg'), star('голубь1.jpg'), star('голубь2.jpg'), star('голубь3.jpg'),
                                        star('голубь4.jpg'), star('голубь5.jpg'), star('голубь6.jpg'), star('голубь7.jpg'),
                                        star('голубь8.jpg'), star('голубь9.jpg'), star('голубь10.jpg'), star('голубь11.jpg'),
                                        star('гончие псы.jpg'), star('гончие псы1.jpg'), star('гончие псы2.jpg'), star('гончие псы3.jpg'),
                                        star('гончие псы4.jpg'), star('гончие псы5.jpg'), star('гончие псы6.jpg'), star('гончие псы7.jpg'),
                                        star('гончие псы8.jpg'), star('гончие псы9.jpg'), star('гончие псы10.jpg'), star('гончие псы11.jpg'),
                                        star('дева.jpg'), star('дева1.jpg'), star('дева2.jpg'), star('дева3.jpg'),
                                        star('дева4.jpg'), star('дева5.jpg'), star('дева6.jpg'), star('дева7.jpg'),
                                        star('дева8.jpg'), star('дева9.jpg'), star('дева10.jpg'), star('дева11.jpg'),
                                        star('дельфин.jpg'), star('дельфин1.jpg'), star('дельфин2.jpg'), star('дельфин3.jpg'),
                                        star('дельфин4.jpg'), star('дельфин5.jpg'), star('дельфин6.jpg'), star('дельфин7.jpg'),
                                        star('дельфин8.jpg'), star('дельфин9.jpg'), star('дельфин10.jpg'), star('дельфин11.jpg'),
                                        star('дракон.jpg'), star('дракон1.jpg'), star('дракон2.jpg'), star('дракон3.jpg'),
                                        star('дракон4.jpg'), star('дракон5.jpg'), star('дракон6.jpg'), star('дракон7.jpg'),
                                        star('дракон8.jpg'), star('дракон9.jpg'), star('дракон10.jpg'), star('дракон11.jpg'),
                                        star('единорог.jpg'), star('единорог1.jpg'), star('единорог2.jpg'), star('единорог3.jpg'),
                                        star('единорог4.jpg'), star('единорог5.jpg'), star('единорог6.jpg'), star('единорог7.jpg'),
                                        star('единорог8.jpg'), star('единорог9.jpg'), star('единорог10.jpg'), star('единорог11.jpg'),
                                        star('жертвенник.jpg'), star('жертвенник1.jpg'), star('жертвенник2.jpg'), star('жертвенник3.jpg'),
                                        star('жертвенник4.jpg'), star('жертвенник5.jpg'), star('жертвенник6.jpg'), star('жертвенник7.jpg'),
                                        star('жертвенник8.jpg'), star('жертвенник9.jpg'), star('жертвенник10.jpg'), star('жертвенник11.jpg'),
                                        star('живописец.jpg'), star('живописец1.jpg'), star('живописец2.jpg'), star('живописец3.jpg'),
                                        star('живописец4.jpg'), star('живописец5.jpg'), star('живописец6.jpg'), star('живописец7.jpg'),
                                        star('живописец8.jpg'), star('живописец9.jpg'), star('живописец10.jpg'), star('живописец11.jpg'),
                                        star('жираф.jpg'), star('жираф1.jpg'), star('жираф2.jpg'), star('жираф3.jpg'),
                                        star('жираф4.jpg'), star('жираф5.jpg'), star('жираф6.jpg'), star('жираф7.jpg'),
                                        star('жираф8.jpg'), star('жираф9.jpg'), star('жираф10.jpg'), star('жираф11.jpg'),
                                        star('журавль.jpg'), star('журавль1.jpg'), star('журавль2.jpg'), star('журавль3.jpg'),
                                        star('журавль4.jpg'), star('журавль5.jpg'), star('журавль6.jpg'), star('журавль7.jpg'),
                                        star('журавль8.jpg'), star('журавль9.jpg'), star('журавль10.jpg'), star('журавль11.jpg'),
                                        star('заяц.jpg'), star('заяц1.jpg'), star('заяц2.jpg'), star('заяц3.jpg'),
                                        star('заяц4.jpg'), star('заяц5.jpg'), star('заяц6.jpg'), star('заяц7.jpg'),
                                        star('заяц8.jpg'), star('заяц9.jpg'), star('заяц10.jpg'), star('заяц11.jpg'),
                                        star('змееносец.jpg'), star('змееносец1.jpg'), star('змееносец2.jpg'), star('змееносец3.jpg'),
                                        star('змееносец4.jpg'), star('змееносец5.jpg'), star('змееносец6.jpg'), star('змееносец7.jpg'),
                                        star('змееносец8.jpg'), star('змееносец9.jpg'), star('змееносец10.jpg'), star('змееносец11.jpg'),
                                        star('змея.jpg'), star('змея1.jpg'), star('змея2.jpg'), star('змея3.jpg'),
                                        star('змея4.jpg'), star('змея5.jpg'), star('змея6.jpg'), star('змея7.jpg'),
                                        star('змея8.jpg'), star('змея9.jpg'), star('змея10.jpg'), star('змея11.jpg'),
                                        star('золотая рыба.jpg'), star('золотая рыба1.jpg'), star('золотая рыба2.jpg'), star('золотая рыба3.jpg'),
                                        star('золотая рыба4.jpg'), star('золотая рыба5.jpg'), star('золотая рыба6.jpg'), star('золотая рыба7.jpg'),
                                        star('золотая рыба8.jpg'), star('золотая рыба9.jpg'), star('золотая рыба10.jpg'), star('золотая рыба11.jpg'),
                                        star('индеец.jpg'), star('индеец1.jpg'), star('индеец2.jpg'), star('индеец3.jpg'),
                                        star('индеец4.jpg'), star('индеец5.jpg'), star('индеец6.jpg'), star('индеец7.jpg'),
                                        star('индеец8.jpg'), star('индеец9.jpg'), star('индеец10.jpg'), star('индеец11.jpg'),
                                        star('кассиопея.jpg'), star('кассиопея1.jpg'), star('кассиопея2.jpg'), star('кассиопея3.jpg'),
                                        star('кассиопея4.jpg'), star('кассиопея5.jpg'), star('кассиопея6.jpg'), star('кассиопея7.jpg'),
                                        star('кассиопея8.jpg'), star('кассиопея9.jpg'), star('кассиопея10.jpg'), star('кассиопея11.jpg'),
                                        star('киль.jpg'), star('киль1.jpg'), star('киль2.jpg'), star('киль3.jpg'),
                                        star('киль4.jpg'), star('киль5.jpg'), star('киль6.jpg'), star('киль7.jpg'),
                                        star('киль8.jpg'), star('киль9.jpg'), star('киль10.jpg'), star('киль11.jpg'),
                                        star('кит.jpg'), star('кит1.jpg'), star('кит2.jpg'), star('кит3.jpg'),
                                        star('кит4.jpg'), star('кит5.jpg'), star('кит6.jpg'), star('кит7.jpg'),
                                        star('кит8.jpg'), star('кит9.jpg'), star('кит10.jpg'), star('кит11.jpg'),
                                        star('козерог.jpg'), star('козерог1.jpg'), star('козерог2.jpg'), star('козерог3.jpg'),
                                        star('козерог4.jpg'), star('козерог5.jpg'), star('козерог6.jpg'), star('козерог7.jpg'),
                                        star('козерог8.jpg'), star('козерог9.jpg'), star('козерог10.jpg'), star('козерог11.jpg'),
                                        star('компас.jpg'), star('компас1.jpg'), star('компас2.jpg'), star('компас3.jpg'),
                                        star('компас4.jpg'), star('компас5.jpg'), star('компас6.jpg'), star('компас7.jpg'),
                                        star('компас8.jpg'), star('компас9.jpg'), star('компас10.jpg'), star('компас11.jpg'),
                                        star('корма.jpg'), star('корма1.jpg'), star('корма2.jpg'), star('корма3.jpg'),
                                        star('корма4.jpg'), star('корма5.jpg'), star('корма6.jpg'), star('корма7.jpg'),
                                        star('корма8.jpg'), star('корма9.jpg'), star('корма10.jpg'), star('корма11.jpg'),
                                        star('лебедь.jpg'), star('лебедь1.jpg'), star('лебедь2.jpg'), star('лебедь3.jpg'),
                                        star('лебедь4.jpg'), star('лебедь5.jpg'), star('лебедь6.jpg'), star('лебедь7.jpg'),
                                        star('лебедь8.jpg'), star('лебедь9.jpg'), star('лебедь10.jpg'), star('лебедь11.jpg'),
                                        star('лев.jpg'), star('лев1.jpg'), star('лев2.jpg'), star('лев3.jpg'),
                                        star('лев4.jpg'), star('лев5.jpg'), star('лев6.jpg'), star('лев7.jpg'),
                                        star('лев8.jpg'), star('лев9.jpg'), star('лев10.jpg'), star('лев11.jpg'),
                                        star('летучая рыба.jpg'), star('летучая рыба1.jpg'), star('летучая рыба2.jpg'), star('летучая рыба3.jpg'),
                                        star('летучая рыба4.jpg'), star('летучая рыба5.jpg'), star('летучая рыба6.jpg'), star('летучая рыба7.jpg'),
                                        star('летучая рыба8.jpg'), star('летучая рыба9.jpg'), star('летучая рыба10.jpg'), star('летучая рыба11.jpg'),
                                        star('лира.jpg'), star('лира1.jpg'), star('лира2.jpg'), star('лира3.jpg'),
                                        star('лира4.jpg'), star('лира5.jpg'), star('лира6.jpg'), star('лира7.jpg'),
                                        star('лира8.jpg'), star('лира9.jpg'), star('лира10.jpg'), star('лира11.jpg'),
                                        star('малая медведица.jpg'), star('малая медведица1.jpg'), star('малая медведица2.jpg'), star('малая медведица3.jpg'),
                                        star('малая медведица4.jpg'), star('малая медведица5.jpg'), star('малая медведица6.jpg'), star('малая медведица7.jpg'),
                                        star('малая медведица8.jpg'), star('малая медведица9.jpg'), star('малая медведица10.jpg'), star('малая медведица11.jpg'),
                                        star('малый конь.jpg'), star('малый конь1.jpg'), star('малый конь2.jpg'), star('малый конь3.jpg'),
                                        star('малый конь4.jpg'), star('малый конь5.jpg'), star('малый конь6.jpg'), star('малый конь7.jpg'),
                                        star('малый конь8.jpg'), star('малый конь9.jpg'), star('малый конь10.jpg'), star('малый конь11.jpg'),
                                        star('малый пес.jpg'), star('малый пес1.jpg'), star('малый пес2.jpg'), star('малый пес3.jpg'),
                                        star('малый пес4.jpg'), star('малый пес5.jpg'), star('малый пес6.jpg'), star('малый пес7.jpg'),
                                        star('малый пес8.jpg'), star('малый пес9.jpg'), star('малый пес10.jpg'), star('малый пес11.jpg'),
                                        star('микроскоп.jpg'), star('микроскоп1.jpg'), star('микроскоп2.jpg'), star('микроскоп3.jpg'),
                                        star('микроскоп4.jpg'), star('микроскоп5.jpg'), star('микроскоп6.jpg'), star('микроскоп7.jpg'),
                                        star('микроскоп8.jpg'), star('микроскоп9.jpg'), star('микроскоп10.jpg'), star('микроскоп11.jpg'),
                                        star('муха.jpg'), star('муха1.jpg'), star('муха2.jpg'), star('муха3.jpg'),
                                        star('муха4.jpg'), star('муха5.jpg'), star('муха6.jpg'), star('муха7.jpg'),
                                        star('муха8.jpg'), star('муха9.jpg'), star('муха10.jpg'), star('муха11.jpg'),
                                        star('насос.jpg'), star('насос1.jpg'), star('насос2.jpg'), star('насос3.jpg'),
                                        star('насос4.jpg'), star('насос5.jpg'), star('насос6.jpg'), star('насос7.jpg'),
                                        star('насос8.jpg'), star('насос9.jpg'), star('насос10.jpg'), star('насос11.jpg'),
                                        star('наугольник.jpg'), star('наугольник1.jpg'), star('наугольник2.jpg'), star('наугольник3.jpg'),
                                        star('наугольник4.jpg'), star('наугольник5.jpg'), star('наугольник6.jpg'), star('наугольник7.jpg'),
                                        star('наугольник8.jpg'), star('наугольник9.jpg'), star('наугольник10.jpg'), star('наугольник11.jpg'),
                                        star('овен.jpg'), star('овен1.jpg'), star('овен2.jpg'), star('овен3.jpg'),
                                        star('овен4.jpg'), star('овен5.jpg'), star('овен6.jpg'), star('овен7.jpg'),
                                        star('овен8.jpg'), star('овен9.jpg'), star('овен10.jpg'), star('овен11.jpg'),
                                        star('октант.jpg'), star('октант1.jpg'), star('октант2.jpg'), star('октант3.jpg'),
                                        star('октант4.jpg'), star('октант5.jpg'), star('октант6.jpg'), star('октант7.jpg'),
                                        star('октант8.jpg'), star('октант9.jpg'), star('октант10.jpg'), star('октант11.jpg'),
                                        star('орел.jpg'), star('орел1.jpg'), star('орел2.jpg'), star('орел3.jpg'),
                                        star('орел4.jpg'), star('орел5.jpg'), star('орел6.jpg'), star('орел7.jpg'),
                                        star('орел8.jpg'), star('орел9.jpg'), star('орел10.jpg'), star('орел11.jpg'),
                                        star('орион.jpg'), star('орион1.jpg'), star('орион2.jpg'), star('орион3.jpg'),
                                        star('орион4.jpg'), star('орион5.jpg'), star('орион6.jpg'), star('орион7.jpg'),
                                        star('орион8.jpg'), star('орион9.jpg'), star('орион10.jpg'), star('орион11.jpg'),
                                        star('павлин.jpg'), star('павлин1.jpg'), star('павлин2.jpg'), star('павлин3.jpg'),
                                        star('павлин4.jpg'), star('павлин5.jpg'), star('павлин6.jpg'), star('павлин7.jpg'),
                                        star('павлин8.jpg'), star('павлин9.jpg'), star('павлин10.jpg'), star('павлин11.jpg'),
                                        star('паруса.jpg'), star('паруса1.jpg'), star('паруса2.jpg'), star('паруса3.jpg'),
                                        star('паруса4.jpg'), star('паруса5.jpg'), star('паруса6.jpg'), star('паруса7.jpg'),
                                        star('паруса8.jpg'), star('паруса9.jpg'), star('паруса10.jpg'), star('паруса11.jpg'),
                                        star('пегас.jpg'), star('пегас1.jpg'), star('пегас2.jpg'), star('пегас3.jpg'),
                                        star('пегас4.jpg'), star('пегас5.jpg'), star('пегас6.jpg'), star('пегас7.jpg'),
                                        star('пегас8.jpg'), star('пегас9.jpg'), star('пегас10.jpg'), star('пегас11.jpg'),
                                        star('персей.jpg'), star('персей1.jpg'), star('персей2.jpg'), star('персей3.jpg'),
                                        star('персей4.jpg'), star('персей5.jpg'), star('персей6.jpg'), star('персей7.jpg'),
                                        star('персей8.jpg'), star('персей9.jpg'), star('персей10.jpg'), star('персей11.jpg'),
                                        star('печь.jpg'), star('печь1.jpg'), star('печь2.jpg'), star('печь3.jpg'),
                                        star('печь4.jpg'), star('печь5.jpg'), star('печь6.jpg'), star('печь7.jpg'),
                                        star('печь8.jpg'), star('печь9.jpg'), star('печь10.jpg'), star('печь11.jpg'),
                                        star('райская птица.jpg'), star('райская птица1.jpg'), star('райская птица2.jpg'), star('райская птица3.jpg'),
                                        star('райская птица4.jpg'), star('райская птица5.jpg'), star('райская птица6.jpg'), star('райская птица7.jpg'),
                                        star('райская птица8.jpg'), star('райская птица9.jpg'), star('райская птица10.jpg'), star('райская птица11.jpg'),
                                        star('рак.jpg'), star('рак1.jpg'), star('рак2.jpg'), star('рак3.jpg'),
                                        star('рак4.jpg'), star('рак5.jpg'), star('рак6.jpg'), star('рак7.jpg'),
                                        star('рак8.jpg'), star('рак9.jpg'), star('рак10.jpg'), star('рак11.jpg'),
                                        star('рыбы.jpg'), star('рыбы1.jpg'), star('рыбы2.jpg'), star('рыбы3.jpg'),
                                        star('рыбы4.jpg'), star('рыбы5.jpg'), star('рыбы6.jpg'), star('рыбы7.jpg'),
                                        star('рыбы8.jpg'), star('рыбы9.jpg'), star('рыбы10.jpg'), star('рыбы11.jpg'),
                                        star('рысь.jpg'), star('рысь1.jpg'), star('рысь2.jpg'), star('рысь3.jpg'),
                                        star('рысь4.jpg'), star('рысь5.jpg'), star('рысь6.jpg'), star('рысь7.jpg'),
                                        star('рысь8.jpg'), star('рысь9.jpg'), star('рысь10.jpg'), star('рысь11.jpg'),
                                        star('северная корона.jpg'), star('северная корона1.jpg'), star('северная корона2.jpg'), star('северная корона3.jpg'),
                                        star('северная корона4.jpg'), star('северная корона5.jpg'), star('северная корона6.jpg'), star('северная корона7.jpg'),
                                        star('северная корона8.jpg'), star('северная корона9.jpg'), star('северная корона10.jpg'), star('северная корона11.jpg'),
                                        star('секстант.jpg'), star('секстант1.jpg'), star('секстант2.jpg'), star('секстант3.jpg'),
                                        star('секстант4.jpg'), star('секстант5.jpg'), star('секстант6.jpg'), star('секстант7.jpg'),
                                        star('секстант8.jpg'), star('секстант9.jpg'), star('секстант10.jpg'), star('секстант11.jpg'),
                                        star('сетка.jpg'), star('сетка1.jpg'), star('сетка2.jpg'), star('сетка3.jpg'),
                                        star('сетка4.jpg'), star('сетка5.jpg'), star('сетка6.jpg'), star('сетка7.jpg'),
                                        star('сетка8.jpg'), star('сетка9.jpg'), star('сетка10.jpg'), star('сетка11.jpg'),
                                        star('скорпион.jpg'), star('скорпион1.jpg'), star('скорпион2.jpg'), star('скорпион3.jpg'),
                                        star('скорпион4.jpg'), star('скорпион5.jpg'), star('скорпион6.jpg'), star('скорпион7.jpg'),
                                        star('скорпион8.jpg'), star('скорпион9.jpg'), star('скорпион10.jpg'), star('скорпион11.jpg'),
                                        star('скульптор.jpg'), star('скульптор1.jpg'), star('скульптор2.jpg'), star('скульптор3.jpg'),
                                        star('скульптор4.jpg'), star('скульптор5.jpg'), star('скульптор6.jpg'), star('скульптор7.jpg'),
                                        star('скульптор8.jpg'), star('скульптор9.jpg'), star('скульптор10.jpg'), star('скульптор11.jpg'),
                                        star('столовая гора.jpg'), star('столовая гора1.jpg'), star('столовая гора2.jpg'), star('столовая гора3.jpg'),
                                        star('столовая гора4.jpg'), star('столовая гора5.jpg'), star('столовая гора6.jpg'), star('столовая гора7.jpg'),
                                        star('столовая гора8.jpg'), star('столовая гора9.jpg'), star('столовая гора10.jpg'), star('столовая гора11.jpg'),
                                        star('стрела.jpg'), star('стрела1.jpg'), star('стрела2.jpg'), star('стрела3.jpg'),
                                        star('стрела4.jpg'), star('стрела5.jpg'), star('стрела6.jpg'), star('стрела7.jpg'),
                                        star('стрела8.jpg'), star('стрела9.jpg'), star('стрела10.jpg'), star('стрела11.jpg'),
                                        star('стрелец.jpg'), star('стрелец1.jpg'), star('стрелец2.jpg'), star('стрелец3.jpg'),
                                        star('стрелец4.jpg'), star('стрелец5.jpg'), star('стрелец6.jpg'), star('стрелец7.jpg'),
                                        star('стрелец8.jpg'), star('стрелец9.jpg'), star('стрелец10.jpg'), star('стрелец11.jpg'),
                                        star('телескоп.jpg'), star('телескоп1.jpg'), star('телескоп2.jpg'), star('телескоп3.jpg'),
                                        star('телескоп4.jpg'), star('телескоп5.jpg'), star('телескоп6.jpg'), star('телескоп7.jpg'),
                                        star('телескоп8.jpg'), star('телескоп9.jpg'), star('телескоп10.jpg'), star('телескоп11.jpg'),
                                        star('телец.jpg'), star('телец1.jpg'), star('телец2.jpg'), star('телец3.jpg'),
                                        star('телец4.jpg'), star('телец5.jpg'), star('телец6.jpg'), star('телец7.jpg'),
                                        star('телец8.jpg'), star('телец9.jpg'), star('телец10.jpg'), star('телец11.jpg'),
                                        star('треугольник.jpg'), star('треугольник1.jpg'), star('треугольник2.jpg'), star('треугольник3.jpg'),
                                        star('треугольник4.jpg'), star('треугольник5.jpg'), star('треугольник6.jpg'), star('треугольник7.jpg'),
                                        star('треугольник8.jpg'), star('треугольник9.jpg'), star('треугольник10.jpg'), star('треугольник11.jpg'),
                                        star('тукан.jpg'), star('тукан1.jpg'), star('тукан2.jpg'), star('тукан3.jpg'),
                                        star('тукан4.jpg'), star('тукан5.jpg'), star('тукан6.jpg'), star('тукан7.jpg'),
                                        star('тукан8.jpg'), star('тукан9.jpg'), star('тукан10.jpg'), star('тукан11.jpg'),
                                        star('феникс.jpg'), star('феникс1.jpg'), star('феникс2.jpg'), star('феникс3.jpg'),
                                        star('феникс4.jpg'), star('феникс5.jpg'), star('феникс6.jpg'), star('феникс7.jpg'),
                                        star('феникс8.jpg'), star('феникс9.jpg'), star('феникс10.jpg'), star('феникс11.jpg'),
                                        star('хамелеон.jpg'), star('хамелеон1.jpg'), star('хамелеон2.jpg'), star('хамелеон3.jpg'),
                                        star('хамелеон4.jpg'), star('хамелеон5.jpg'), star('хамелеон6.jpg'), star('хамелеон7.jpg'),
                                        star('хамелеон8.jpg'), star('хамелеон9.jpg'), star('хамелеон10.jpg'), star('хамелеон11.jpg'),
                                        star('центавр.jpg'), star('центавр1.jpg'), star('центавр2.jpg'), star('центавр3.jpg'),
                                        star('центавр4.jpg'), star('центавр5.jpg'), star('центавр6.jpg'), star('центавр7.jpg'),
                                        star('центавр8.jpg'), star('центавр9.jpg'), star('центавр10.jpg'), star('центавр11.jpg'),
                                        star('цефей.jpg'), star('цефей1.jpg'), star('цефей2.jpg'), star('цефей3.jpg'),
                                        star('цефей4.jpg'), star('цефей5.jpg'), star('цефей6.jpg'), star('цефей7.jpg'),
                                        star('цефей8.jpg'), star('цефей9.jpg'), star('цефей10.jpg'), star('цефей11.jpg'),
                                        star('циркуль.jpg'), star('циркуль1.jpg'), star('циркуль2.jpg'), star('циркуль3.jpg'),
                                        star('циркуль4.jpg'), star('циркуль5.jpg'), star('циркуль6.jpg'), star('циркуль7.jpg'),
                                        star('циркуль8.jpg'), star('циркуль9.jpg'), star('циркуль10.jpg'), star('циркуль11.jpg'),
                                        star('часы.jpg'), star('часы1.jpg'), star('часы2.jpg'), star('часы3.jpg'),
                                        star('часы4.jpg'), star('часы5.jpg'), star('часы6.jpg'), star('часы7.jpg'),
                                        star('часы8.jpg'), star('часы9.jpg'), star('часы10.jpg'), star('часы11.jpg'),
                                        star('чаша.jpg'), star('чаша1.jpg'), star('чаша2.jpg'), star('чаша3.jpg'),
                                        star('чаша4.jpg'), star('чаша5.jpg'), star('чаша6.jpg'), star('чаша7.jpg'),
                                        star('чаша8.jpg'), star('чаша9.jpg'), star('чаша10.jpg'), star('чаша11.jpg'),
                                        star('щит.jpg'), star('щит1.jpg'), star('щит2.jpg'), star('щит3.jpg'),
                                        star('щит4.jpg'), star('щит5.jpg'), star('щит6.jpg'), star('щит7.jpg'),
                                        star('щит8.jpg'), star('щит9.jpg'), star('щит10.jpg'), star('щит11.jpg'),
                                        star('эридан.jpg'), star('эридан1.jpg'), star('эридан2.jpg'), star('эридан3.jpg'),
                                        star('эридан4.jpg'), star('эридан5.jpg'), star('эридан6.jpg'), star('эридан7.jpg'),
                                        star('эридан8.jpg'), star('эридан9.jpg'), star('эридан10.jpg'), star('эридан11.jpg'),
                                        star('южная гидра.jpg'), star('южная гидра1.jpg'), star('южная гидра2.jpg'), star('южная гидра3.jpg'),
                                        star('южная гидра4.jpg'), star('южная гидра5.jpg'), star('южная гидра6.jpg'), star('южная гидра7.jpg'),
                                        star('южная гидра8.jpg'), star('южная гидра9.jpg'), star('южная гидра10.jpg'), star('южная гидра11.jpg'),
                                        star('южная корона.jpg'), star('южная корона1.jpg'), star('южная корона2.jpg'), star('южная корона3.jpg'),
                                        star('южная корона4.jpg'), star('южная корона5.jpg'), star('южная корона6.jpg'), star('южная корона7.jpg'),
                                        star('южная корона8.jpg'), star('южная корона9.jpg'), star('южная корона10.jpg'), star('южная корона11.jpg'),
                                        star('южная рыба.jpg'), star('южная рыба1.jpg'), star('южная рыба2.jpg'), star('южная рыба3.jpg'),
                                        star('южная рыба4.jpg'), star('южная рыба5.jpg'), star('южная рыба6.jpg'), star('южная рыба7.jpg'),
                                        star('южная рыба8.jpg'), star('южная рыба9.jpg'), star('южная рыба10.jpg'), star('южная рыба11.jpg'),
                                        star('южный крест.jpg'), star('южный крест1.jpg'), star('южный крест2.jpg'), star('южный крест3.jpg'),
                                        star('южный крест4.jpg'), star('южный крест5.jpg'), star('южный крест6.jpg'), star('южный крест7.jpg'),
                                        star('южный крест8.jpg'), star('южный крест9.jpg'), star('южный крест10.jpg'), star('южный крест11.jpg'),
                                        star('южный треугольник.jpg'), star('южный треугольник1.jpg'), star('южный треугольник2.jpg'), star('южный треугольник3.jpg'),
                                        star('южный треугольник4.jpg'), star('южный треугольник5.jpg'), star('южный треугольник6.jpg'), star('южный треугольник7.jpg'),
                                        star('южный треугольник8.jpg'), star('южный треугольник9.jpg'), star('южный треугольник10.jpg'), star('южный треугольник11.jpg'),
                                        star('ящерица.jpg'), star('ящерица1.jpg'), star('ящерица2.jpg'), star('ящерица3.jpg'),
                                        star('ящерица4.jpg'), star('ящерица5.jpg'), star('ящерица6.jpg'), star('ящерица7.jpg'),
                                        star('ящерица8.jpg'), star('ящерица9.jpg'), star('ящерица10.jpg'), star('ящерица11.jpg')])}}


with open('db_stars.pkl', 'wb') as stars_pickle:
    pickle.dump(data, stars_pickle)
