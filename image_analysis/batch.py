from lmdeploy import pipeline, PytorchEngineConfig, VisionConfig
from lmdeploy.messages import GenerationConfig, PytorchEngineConfig
from lmdeploy.vl import load_image
from lmdeploy.vl.utils import encode_image_base64
from utils.markdown import extract_json_objects

import datetime

# Please set tp=2 for the 38B version and tp=8 for the 241B-A28B version.
model = "OpenGVLab/InternVL3_5-8B"
pipe = pipeline(
    model,
    backend_config=PytorchEngineConfig(session_len=8192, tp=1),
    vision_config=VisionConfig(max_batch_size=8),
)

# backend_config=TurbomindEngineConfig(
#     max_batch_size=32,
#     enable_prefix_caching=True,
#     cache_max_entry_count=0.8,
#     session_len=8192,
# ))

screenshot_format = {
    'type': 'object',
    'properties': {
        'filter_category': { 'type': 'string' },
        'title': { 'type': 'string' },
        'summary': { 'type': 'string' },
        'keywords': { 'type': 'string' },
        'dewey_classification': { 'type': 'array', 'items': { 'type': 'string' } },        
    },
    'required': ['filter_category']
}
gen_config = GenerationConfig(response_format=dict(type='json_schema', json_schema=dict(name='screenshot', schema=screenshot_format)))

image_urls = [
    # "https://webarchiving.internetarchive.eu/screenshots/main-20250316/test/test.jpg",
    # "https://webarchiving.internetarchive.eu/screenshots/main-20250316/test/test1.jpg",
    # "https://webarchiving.internetarchive.eu/screenshots/main-20250316/test/test1.jpg",
    # "https://webarchiving.internetarchive.eu/iiif/2/collages!main-20250316!row-2.tif/142064,20480,2732,2048/!800,600/0/default.jpg"
    # "https://webarchiving.internetarchive.eu/iiif/2/collages!nl-20250316!row-2.tif/133868,20480,2732,2048/!800,600/0/default.jpg",
    # "https://webarchiving.internetarchive.eu/iiif/2/collages!main-20250316!row-2.tif/139332,24576,2732,2048/!800,600/0/default.jpg",
    # "https://webarchiving.internetarchive.eu/iiif/2/collages!main-20250316!row-2.tif/762228,34816,2732,2048/!800,600/0/default.jpg",
    # "https://webarchiving.internetarchive.eu/iiif/2/collages!main-20250316!row-2.tif/196704,12288,2732,2048/!800,600/0/default.jpg",
    # "https://webarchiving.internetarchive.eu/iiif/2/collages!main-20250316!row-0.tif/620164,6144,2732,2048/!800,600/0/default.jpg",    
    # "https://webarchiving.internetarchive.eu/iiif/2/collages!main-20250316!row-8.tif/631092,6144,2732,2048/!800,600/0/default.jpg"
    
    "https://webarchiving.internetarchive.eu/iiif/2/collages!main-20250316!row-0.tif/2732,0,2732,2048/!800,600/0/default.jpg#filename=http%3A%2F%2Fwww.ebaystores.nl%2F",
    "https://webarchiving.internetarchive.eu/iiif/2/collages!main-20250316!row-0.tif/0,0,2732,2048/!800,600/0/default.jpg#filename=http%3A%2F%2Fwww.genealogy.nl%2F",
    "https://webarchiving.internetarchive.eu/iiif/2/collages!main-20250316!row-0.tif/5464,0,2732,2048/!800,600/0/default.jpg#filename=http%3A%2F%2Fwww.aqualine.nl%2F",
    "https://webarchiving.internetarchive.eu/iiif/2/collages!main-20250316!row-0.tif/8196,0,2732,2048/!800,600/0/default.jpg#filename=http%3A%2F%2Fwww.lokatienet.nl%2F",
    "https://webarchiving.internetarchive.eu/iiif/2/collages!main-20250316!row-0.tif/10928,0,2732,2048/!800,600/0/default.jpg#filename=http%3A%2F%2Fwww.sun.nl%2F",
    "https://webarchiving.internetarchive.eu/iiif/2/collages!main-20250316!row-0.tif/13660,0,2732,2048/!800,600/0/default.jpg#filename=http%3A%2F%2Fwww.azr.nl%2F",
    "https://webarchiving.internetarchive.eu/iiif/2/collages!main-20250316!row-0.tif/16392,0,2732,2048/!800,600/0/default.jpg#filename=http%3A%2F%2Fwww.amev.nl%2F",
    "https://webarchiving.internetarchive.eu/iiif/2/collages!main-20250316!row-0.tif/19124,0,2732,2048/!800,600/0/default.jpg#filename=http%3A%2F%2Fwww.basf.nl%2F",
    "https://webarchiving.internetarchive.eu/iiif/2/collages!main-20250316!row-0.tif/21856,0,2732,2048/!800,600/0/default.jpg#filename=http%3A%2F%2Fwww.urbanspots.nl%2F",
    "https://webarchiving.internetarchive.eu/iiif/2/collages!main-20250316!row-0.tif/24588,0,2732,2048/!800,600/0/default.jpg#filename=http%3A%2F%2Fwww.porsche-914-club-holland.nl%2F",
    "https://webarchiving.internetarchive.eu/iiif/2/collages!main-20250316!row-0.tif/27320,0,2732,2048/!800,600/0/default.jpg#filename=http%3A%2F%2Fwww.beurskenners.nl%2F",
    "https://webarchiving.internetarchive.eu/iiif/2/collages!main-20250316!row-0.tif/30052,0,2732,2048/!800,600/0/default.jpg#filename=http%3A%2F%2Fwww.abnamro.nl%2F",
    "https://webarchiving.internetarchive.eu/iiif/2/collages!main-20250316!row-0.tif/32784,0,2732,2048/!800,600/0/default.jpg#filename=http%3A%2F%2Fwww.meubelstoffenshop.nl%2F",
    "https://webarchiving.internetarchive.eu/iiif/2/collages!main-20250316!row-0.tif/35516,0,2732,2048/!800,600/0/default.jpg#filename=http%3A%2F%2Fwww.fiscanet.nl%2F",
    "https://webarchiving.internetarchive.eu/iiif/2/collages!main-20250316!row-0.tif/38248,0,2732,2048/!800,600/0/default.jpg#filename=http%3A%2F%2Faanmelder.nl%2F",
    "https://webarchiving.internetarchive.eu/iiif/2/collages!main-20250316!row-0.tif/40980,0,2732,2048/!800,600/0/default.jpg#filename=http%3A%2F%2Fwww.overwoerden.nl%2F",
    "https://webarchiving.internetarchive.eu/iiif/2/collages!main-20250316!row-0.tif/43712,0,2732,2048/!800,600/0/default.jpg#filename=http%3A%2F%2Fwww.youtulip.nl%2F",
    "https://webarchiving.internetarchive.eu/iiif/2/collages!main-20250316!row-0.tif/46444,0,2732,2048/!800,600/0/default.jpg#filename=http%3A%2F%2Fwww.rwe.nl%2F",
    "https://webarchiving.internetarchive.eu/iiif/2/collages!main-20250316!row-0.tif/49176,0,2732,2048/!800,600/0/default.jpg#filename=http%3A%2F%2Fwww.meespierson.nl%2F",
    "https://webarchiving.internetarchive.eu/iiif/2/collages!main-20250316!row-0.tif/51908,0,2732,2048/!800,600/0/default.jpg#filename=http%3A%2F%2Fwww.funmetgsm.nl%2F",
    "https://webarchiving.internetarchive.eu/iiif/2/collages!main-20250316!row-0.tif/54640,0,2732,2048/!800,600/0/default.jpg#filename=http%3A%2F%2Fwww.skeelerspel.nl%2F",
    "https://webarchiving.internetarchive.eu/iiif/2/collages!main-20250316!row-0.tif/57372,0,2732,2048/!800,600/0/default.jpg#filename=http%3A%2F%2Fwww.verlanglijstje.nl%2F",
    "https://webarchiving.internetarchive.eu/iiif/2/collages!main-20250316!row-0.tif/60104,0,2732,2048/!800,600/0/default.jpg#filename=http%3A%2F%2Fwww.kwaliteitsapotheek.nl%2F",
    "https://webarchiving.internetarchive.eu/iiif/2/collages!main-20250316!row-0.tif/62836,0,2732,2048/!800,600/0/default.jpg#filename=http%3A%2F%2Fwww.porscheclub928.nl%2F",
    "https://webarchiving.internetarchive.eu/iiif/2/collages!main-20250316!row-0.tif/65568,0,2732,2048/!800,600/0/default.jpg#filename=http%3A%2F%2Fwww.hastec.nl%2F",
    "https://webarchiving.internetarchive.eu/iiif/2/collages!main-20250316!row-0.tif/68300,0,2732,2048/!800,600/0/default.jpg#filename=http%3A%2F%2Fwww.alh.nl%2F",
    "https://webarchiving.internetarchive.eu/iiif/2/collages!main-20250316!row-0.tif/71032,0,2732,2048/!800,600/0/default.jpg#filename=http%3A%2F%2Fwww.speedlinq.nl%2F",
    "https://webarchiving.internetarchive.eu/iiif/2/collages!main-20250316!row-0.tif/73764,0,2732,2048/!800,600/0/default.jpg#filename=http%3A%2F%2Fwww.orange.nl%2F",
    "https://webarchiving.internetarchive.eu/iiif/2/collages!main-20250316!row-0.tif/76496,0,2732,2048/!800,600/0/default.jpg#filename=http%3A%2F%2Fwww.teamlink.nl%2F",
    "https://webarchiving.internetarchive.eu/iiif/2/collages!main-20250316!row-0.tif/79228,0,2732,2048/!800,600/0/default.jpg#filename=http%3A%2F%2Fwww.hslzuid.nl%2F",
    "https://webarchiving.internetarchive.eu/iiif/2/collages!main-20250316!row-0.tif/81960,0,2732,2048/!800,600/0/default.jpg#filename=http%3A%2F%2Fmaxdigital.nl%2F",
    "https://webarchiving.internetarchive.eu/iiif/2/collages!main-20250316!row-0.tif/84692,0,2732,2048/!800,600/0/default.jpg#filename=http%3A%2F%2Fwww.wweholland.nl%2F",
    "https://webarchiving.internetarchive.eu/iiif/2/collages!main-20250316!row-0.tif/87424,0,2732,2048/!800,600/0/default.jpg#filename=http%3A%2F%2Fwww.att.nl%2F",
    "https://webarchiving.internetarchive.eu/iiif/2/collages!main-20250316!row-0.tif/90156,0,2732,2048/!800,600/0/default.jpg#filename=http%3A%2F%2Fwww.automotivecenter.nl%2F",
    "https://webarchiving.internetarchive.eu/iiif/2/collages!main-20250316!row-0.tif/92888,0,2732,2048/!800,600/0/default.jpg#filename=http%3A%2F%2Fautovisieblog.nl%2F",
    "https://webarchiving.internetarchive.eu/iiif/2/collages!main-20250316!row-0.tif/95620,0,2732,2048/!800,600/0/default.jpg#filename=http%3A%2F%2Finternetinzicht.nl%2F",
    "https://webarchiving.internetarchive.eu/iiif/2/collages!main-20250316!row-0.tif/98352,0,2732,2048/!800,600/0/default.jpg#filename=http%3A%2F%2Fgsm-cover.nl%2F",
    "https://webarchiving.internetarchive.eu/iiif/2/collages!main-20250316!row-0.tif/101084,0,2732,2048/!800,600/0/default.jpg#filename=http%3A%2F%2Flange-mensen.nl%2F",
    "https://webarchiving.internetarchive.eu/iiif/2/collages!main-20250316!row-0.tif/103816,0,2732,2048/!800,600/0/default.jpg#filename=http%3A%2F%2Fprivate.nl%2F",
    "https://webarchiving.internetarchive.eu/iiif/2/collages!main-20250316!row-0.tif/106548,0,2732,2048/!800,600/0/default.jpg#filename=http%3A%2F%2Fnattevoegen.nl%2F",
    "https://webarchiving.internetarchive.eu/iiif/2/collages!main-20250316!row-0.tif/109280,0,2732,2048/!800,600/0/default.jpg#filename=http%3A%2F%2Fwww.belzaak.nl%2F",
    "https://webarchiving.internetarchive.eu/iiif/2/collages!main-20250316!row-0.tif/112012,0,2732,2048/!800,600/0/default.jpg#filename=http%3A%2F%2Fwww.fastminute.nl%2F",
    "https://webarchiving.internetarchive.eu/iiif/2/collages!main-20250316!row-0.tif/114744,0,2732,2048/!800,600/0/default.jpg#filename=http%3A%2F%2Fwww.tntpost.nl%2F%2F",
    "https://webarchiving.internetarchive.eu/iiif/2/collages!main-20250316!row-0.tif/117476,0,2732,2048/!800,600/0/default.jpg#filename=http%3A%2F%2Fwww.eenveiligamsterdam.nl%2F",
    "https://webarchiving.internetarchive.eu/iiif/2/collages!main-20250316!row-0.tif/120208,0,2732,2048/!800,600/0/default.jpg#filename=http%3A%2F%2Fwww.dactylo.nl%2F",
    "https://webarchiving.internetarchive.eu/iiif/2/collages!main-20250316!row-0.tif/122940,0,2732,2048/!800,600/0/default.jpg#filename=http%3A%2F%2Fwww.sepfeestartikelen.nl%2F",
    "https://webarchiving.internetarchive.eu/iiif/2/collages!main-20250316!row-0.tif/125672,0,2732,2048/!800,600/0/default.jpg#filename=http%3A%2F%2Fwww.fotografieleenkoper.nl%2F",
    "https://webarchiving.internetarchive.eu/iiif/2/collages!main-20250316!row-0.tif/128404,0,2732,2048/!800,600/0/default.jpg#filename=http%3A%2F%2Fwww.henkdevelde.nl%2F",
    "https://webarchiving.internetarchive.eu/iiif/2/collages!main-20250316!row-0.tif/131136,0,2732,2048/!800,600/0/default.jpg#filename=http%3A%2F%2Fzondermakelaar.nl%2F",
    "https://webarchiving.internetarchive.eu/iiif/2/collages!main-20250316!row-0.tif/133868,0,2732,2048/!800,600/0/default.jpg#filename=http%3A%2F%2Fnhdp.nl%2F",
    "https://webarchiving.internetarchive.eu/iiif/2/collages!main-20250316!row-0.tif/136600,0,2732,2048/!800,600/0/default.jpg#filename=http%3A%2F%2Fhhh.nl%2F",
    "https://webarchiving.internetarchive.eu/iiif/2/collages!main-20250316!row-0.tif/139332,0,2732,2048/!800,600/0/default.jpg#filename=http%3A%2F%2Fwww.fishersci.nl%2F",
    "https://webarchiving.internetarchive.eu/iiif/2/collages!main-20250316!row-0.tif/142064,0,2732,2048/!800,600/0/default.jpg#filename=http%3A%2F%2Fwww.pixagogo.nl%2F",
    "https://webarchiving.internetarchive.eu/iiif/2/collages!main-20250316!row-0.tif/144796,0,2732,2048/!800,600/0/default.jpg#filename=http%3A%2F%2Fwww.nhd.nl%2F",
    "https://webarchiving.internetarchive.eu/iiif/2/collages!main-20250316!row-0.tif/147528,0,2732,2048/!800,600/0/default.jpg#filename=http%3A%2F%2Fwww.internetadressengids.nl%2F",
    "https://webarchiving.internetarchive.eu/iiif/2/collages!main-20250316!row-0.tif/150260,0,2732,2048/!800,600/0/default.jpg#filename=http%3A%2F%2Fwww.smartdigital.nl%2F",
    "https://webarchiving.internetarchive.eu/iiif/2/collages!main-20250316!row-0.tif/152992,0,2732,2048/!800,600/0/default.jpg#filename=http%3A%2F%2Fmoooishop.nl%2F",
    "https://webarchiving.internetarchive.eu/iiif/2/collages!main-20250316!row-0.tif/155724,0,2732,2048/!800,600/0/default.jpg#filename=http%3A%2F%2Fstuwrotterdam.nl%2F",
    "https://webarchiving.internetarchive.eu/iiif/2/collages!main-20250316!row-0.tif/158456,0,2732,2048/!800,600/0/default.jpg#filename=http%3A%2F%2Fwww.goldentulip.nl%2F",
    "https://webarchiving.internetarchive.eu/iiif/2/collages!main-20250316!row-0.tif/161188,0,2732,2048/!800,600/0/default.jpg#filename=http%3A%2F%2Fwww.netdesign.nl%2F",
    "https://webarchiving.internetarchive.eu/iiif/2/collages!main-20250316!row-0.tif/163920,0,2732,2048/!800,600/0/default.jpg#filename=http%3A%2F%2Fpersvraag.nl%2F",
    "https://webarchiving.internetarchive.eu/iiif/2/collages!main-20250316!row-0.tif/166652,0,2732,2048/!800,600/0/default.jpg#filename=http%3A%2F%2Fdordrecht.nl%2F",
    "https://webarchiving.internetarchive.eu/iiif/2/collages!main-20250316!row-0.tif/169384,0,2732,2048/!800,600/0/default.jpg#filename=http%3A%2F%2Fwww.movia.nl%2F",
    "https://webarchiving.internetarchive.eu/iiif/2/collages!main-20250316!row-0.tif/172116,0,2732,2048/!800,600/0/default.jpg#filename=http%3A%2F%2Fwww.jumbosupermarkten.nl%2F",
    "https://webarchiving.internetarchive.eu/iiif/2/collages!main-20250316!row-0.tif/174848,0,2732,2048/!800,600/0/default.jpg#filename=http%3A%2F%2Fwww.tpgpostbusiness.nl%2F%2F",
    "https://webarchiving.internetarchive.eu/iiif/2/collages!main-20250316!row-0.tif/177580,0,2732,2048/!800,600/0/default.jpg#filename=http%3A%2F%2Fwww.klimaatcomfort.nl%2F",
    "https://webarchiving.internetarchive.eu/iiif/2/collages!main-20250316!row-0.tif/180312,0,2732,2048/!800,600/0/default.jpg#filename=http%3A%2F%2Fwww.collectiblez.nl%2F",
    "https://webarchiving.internetarchive.eu/iiif/2/collages!main-20250316!row-0.tif/183044,0,2732,2048/!800,600/0/default.jpg#filename=http%3A%2F%2Fhetechat.nl%2F",
    "https://webarchiving.internetarchive.eu/iiif/2/collages!main-20250316!row-0.tif/185776,0,2732,2048/!800,600/0/default.jpg#filename=http%3A%2F%2Fjuncker.nl%2F",
    "https://webarchiving.internetarchive.eu/iiif/2/collages!main-20250316!row-0.tif/188508,0,2732,2048/!800,600/0/default.jpg#filename=http%3A%2F%2Fkortingplaza.nl%2F",
    "https://webarchiving.internetarchive.eu/iiif/2/collages!main-20250316!row-0.tif/191240,0,2732,2048/!800,600/0/default.jpg#filename=http%3A%2F%2Fpathedekuip.nl%2F",
    "https://webarchiving.internetarchive.eu/iiif/2/collages!main-20250316!row-0.tif/193972,0,2732,2048/!800,600/0/default.jpg#filename=http%3A%2F%2Fperfectlyyou.nl%2F",
    "https://webarchiving.internetarchive.eu/iiif/2/collages!main-20250316!row-0.tif/196704,0,2732,2048/!800,600/0/default.jpg#filename=http%3A%2F%2Fwww.saab.nl%2F",
    "https://webarchiving.internetarchive.eu/iiif/2/collages!main-20250316!row-0.tif/199436,0,2732,2048/!800,600/0/default.jpg#filename=http%3A%2F%2Fwww.huisdierdroom.nl%2F",
    "https://webarchiving.internetarchive.eu/iiif/2/collages!main-20250316!row-0.tif/202168,0,2732,2048/!800,600/0/default.jpg#filename=http%3A%2F%2Fwww.trustpilot.nl%2F",
    "https://webarchiving.internetarchive.eu/iiif/2/collages!main-20250316!row-0.tif/204900,0,2732,2048/!800,600/0/default.jpg#filename=http%3A%2F%2Fwww.dfds.nl%2F",
    "https://webarchiving.internetarchive.eu/iiif/2/collages!main-20250316!row-0.tif/207632,0,2732,2048/!800,600/0/default.jpg#filename=http%3A%2F%2Fwww.deltalloyd.nl%2F",
    "https://webarchiving.internetarchive.eu/iiif/2/collages!main-20250316!row-0.tif/210364,0,2732,2048/!800,600/0/default.jpg#filename=http%3A%2F%2Fwww.sporen.nl%2F",
    "https://webarchiving.internetarchive.eu/iiif/2/collages!main-20250316!row-0.tif/213096,0,2732,2048/!800,600/0/default.jpg#filename=http%3A%2F%2Fwww.rentino-rio.nl%2F",
    "https://webarchiving.internetarchive.eu/iiif/2/collages!main-20250316!row-0.tif/215828,0,2732,2048/!800,600/0/default.jpg#filename=http%3A%2F%2Fautomic.nl%2F",
    "https://webarchiving.internetarchive.eu/iiif/2/collages!main-20250316!row-0.tif/218560,0,2732,2048/!800,600/0/default.jpg#filename=http%3A%2F%2Fbex.nl%2F",
    "https://webarchiving.internetarchive.eu/iiif/2/collages!main-20250316!row-0.tif/221292,0,2732,2048/!800,600/0/default.jpg#filename=http%3A%2F%2Fautoverkopentips.nl%2F",
    "https://webarchiving.internetarchive.eu/iiif/2/collages!main-20250316!row-0.tif/224024,0,2732,2048/!800,600/0/default.jpg#filename=http%3A%2F%2Fhotelplan.nl%2F",
    "https://webarchiving.internetarchive.eu/iiif/2/collages!main-20250316!row-0.tif/226756,0,2732,2048/!800,600/0/default.jpg#filename=http%3A%2F%2Fwww.hro.nl%2F",
    "https://webarchiving.internetarchive.eu/iiif/2/collages!main-20250316!row-0.tif/229488,0,2732,2048/!800,600/0/default.jpg#filename=http%3A%2F%2Fwww.hes-rdam.nl%2F",
    "https://webarchiving.internetarchive.eu/iiif/2/collages!main-20250316!row-0.tif/232220,0,2732,2048/!800,600/0/default.jpg#filename=http%3A%2F%2Fwww.wetwegwijzer.nl%2F",
    "https://webarchiving.internetarchive.eu/iiif/2/collages!main-20250316!row-0.tif/234952,0,2732,2048/!800,600/0/default.jpg#filename=http%3A%2F%2Fwww.tuschinski.nl%2F",
    "https://webarchiving.internetarchive.eu/iiif/2/collages!main-20250316!row-0.tif/237684,0,2732,2048/!800,600/0/default.jpg#filename=http%3A%2F%2Fwww.hogeschool-rotterdam.nl%2F",
    "https://webarchiving.internetarchive.eu/iiif/2/collages!main-20250316!row-0.tif/240416,0,2732,2048/!800,600/0/default.jpg#filename=http%3A%2F%2Fq-music.nl%2F",
    "https://webarchiving.internetarchive.eu/iiif/2/collages!main-20250316!row-0.tif/243148,0,2732,2048/!800,600/0/default.jpg#filename=http%3A%2F%2Fgamecloud.nl%2F",
    "https://webarchiving.internetarchive.eu/iiif/2/collages!main-20250316!row-0.tif/245880,0,2732,2048/!800,600/0/default.jpg#filename=http%3A%2F%2Fwww.hr.nl%2F",
    "https://webarchiving.internetarchive.eu/iiif/2/collages!main-20250316!row-0.tif/248612,0,2732,2048/!800,600/0/default.jpg#filename=http%3A%2F%2Fnoborder.nl%2F",
    "https://webarchiving.internetarchive.eu/iiif/2/collages!main-20250316!row-0.tif/251344,0,2732,2048/!800,600/0/default.jpg#filename=http%3A%2F%2Fstoelvanklaveren.nl%2F",
    "https://webarchiving.internetarchive.eu/iiif/2/collages!main-20250316!row-0.tif/254076,0,2732,2048/!800,600/0/default.jpg#filename=http%3A%2F%2Fbbnbouwmaterialen.nl%2F",
    "https://webarchiving.internetarchive.eu/iiif/2/collages!main-20250316!row-0.tif/256808,0,2732,2048/!800,600/0/default.jpg#filename=http%3A%2F%2Fwww.seatonly.nl%2F",
    "https://webarchiving.internetarchive.eu/iiif/2/collages!main-20250316!row-0.tif/259540,0,2732,2048/!800,600/0/default.jpg#filename=http%3A%2F%2Fjaguar.nl%2F",
    "https://webarchiving.internetarchive.eu/iiif/2/collages!main-20250316!row-0.tif/262272,0,2732,2048/!800,600/0/default.jpg#filename=http%3A%2F%2Fshavershop.nl%2F",
    "https://webarchiving.internetarchive.eu/iiif/2/collages!main-20250316!row-0.tif/265004,0,2732,2048/!800,600/0/default.jpg#filename=http%3A%2F%2Fgrandorado.nl%2F",
    "https://webarchiving.internetarchive.eu/iiif/2/collages!main-20250316!row-0.tif/267736,0,2732,2048/!800,600/0/default.jpg#filename=http%3A%2F%2Fstartrekvoyager.nl%2F",
    "https://webarchiving.internetarchive.eu/iiif/2/collages!main-20250316!row-0.tif/270468,0,2732,2048/!800,600/0/default.jpg#filename=http%3A%2F%2Fautobloggers.nl%2F",
]

print(datetime.datetime.now())
prompts = [
    (
        """
        Extract content about the screenshot provided in JSON format.
        Always follow the rules:
        - be extra precise when reading company and product names

        JSON Fields:
        - filter_category: ok, parked or newly registered domain or placeholder pages or page from domain registrars, pornographic or adult-only content (including sex toys and sex shops), 404. Use values: placeholder | adult | valid
        - title: translated to english
        - summary (english)
        - keywords (english)
        - dewey_classification
        """,
        load_image(img_url),
    )
    for img_url in image_urls
]
#response = pipe(prompts, gen_config=gen_config)
response = pipe(prompts)

print(datetime.datetime.now())

for i, res in enumerate(response):
    json = extract_json_objects(res.text)[0]    
    json['id'] = image_urls[i]
    print(json)

