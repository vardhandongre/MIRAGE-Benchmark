import sys
sys.path.append('../../')
from chat_models.OpenAI_Chat import GPT4O
from pydantic import BaseModel
import json
import multiprocessing
import os
from tqdm import tqdm
import argparse
from typing import List


# Define a new Pydantic model for the direct answer check
class TAGS(BaseModel):
    image_description: str
    management_instructions: str
    miscellaneous_facts: List[str]
    
    def to_json(self):
        return {"image_description": self.image_description, "management_instructions": self.management_instructions, "miscellaneous_facts": self.miscellaneous_facts}

class TAGS2(BaseModel):
    symptom_description: str
    management_instructions: str
    miscellaneous_facts: List[str]
    
    def to_json(self):
        return {"symptom_description": self.symptom_description, "management_instructions": self.management_instructions, "miscellaneous_facts": self.miscellaneous_facts}

class TAGS3(BaseModel):
    image_description: str
    miscellaneous_facts: List[str]
    
    def to_json(self):
        return {"image_description": self.image_description, "miscellaneous_facts": self.miscellaneous_facts}

class TAGS4(BaseModel):
    symptom_description: str
    miscellaneous_facts: List[str]
    
    def to_json(self):
        return {"symptom_description": self.symptom_description, "miscellaneous_facts": self.miscellaneous_facts}



class Fact_Extractor:
    def __init__(self, raw_data_file, output_file, model_name="gpt-4o-mini", num_processes=None):
        self.raw_data_file = raw_data_file
        self.output_file = output_file
        self.model_name = model_name
        self.num_processes = num_processes if num_processes is not None else os.cpu_count()

    # Prompts for Weeds/Invasive Plants Management
    def get_prompt_wipm(self, item):        
        prefix = """You are an assistant whose job it is to extract categories of information from a weeds/invasive plants management Q&A. \
These categories are "image description", "management instructions", and "miscellaneous facts". Make sure that "image \
description" describe the visual qualities that can be referenced in the hypothetical image. \
If there is no mention of an image description in the conversation, output 'none'. \
Ensure that miscellaneous facts are independent of the exact situation and are self-contained atomic facts, meaning that each fact is a single coherent entity.

Format the response as a json. If no information for specific category is present, output "none".\n\n"""

        example = """<Example1>
Title: Invasive grass flower
User Question: This small white flower invades the grass and multiplies and takes over the grass. It flowered in May but now in June stopped flowering. It roots are like a sweet potatoes and difficult to remove, you need to dig it up making your lawn look patchy.
Expert Answer: The weed you are trying to rid your lawn of is pennywort (Hydrocotyle americana, sometimes also known as dollarweed. It spreads by seed and by underground rhizomes and is a perennial that blooms early. It thrives in moist areas. The best way to control this broadleaf weed is to maintain a healthy lawn by regularly mowing at the recommended height for your variety of grass and watering deeply and infrequently to encourage deeper root growth. Monitor your lawn for areas that may need improved soil drainage. Fertilize your lawn appropriately; as recommended for your type of grass. Remove the weeds you presently have by hand pulling making sure to remove all roots. If your infestation is too broad to control by cultural methods, chemical control options are available. Use a herbicide designed to target this specific type of weed. Your local nursery operator can help you select the most effective application. When using any chemical read the label thoroughly and follow the instructions provided regarding the proper use and disposal Thank you for your question.

Model Response: 
{'image_description': 'small white flower with roots like sweet potato.', 'management_instructions': 'Maintain a \
healthy lawn by regularly mowing at the recommended height for your variety of grass and watering deeply and \
infrequently to encourage deeper root growth. Monitor your lawn for areas that may need improved soil drainage. \
Fertilize your lawn appropriately; as recommended for your type of grass. Remove the weeds you presently have by \
hand pulling making sure to remove all roots. If your infestation is too broad to control by cultural methods, \
chemical control options are available. Use a herbicide designed to target this specific type of weed', \
'miscellaneous_facts': ['pennywort spreads by seed and by underground rhizomes', 'pennywort is a perennial that \
blooms early', 'pennywort thrives in moist areas.']}
</Example1>


<Example2>
Title: Is this horsenettle?
User Question: I am thinking that the attached photo is of horsenettle. Is that correct? I would prefer not to use any chemicals, but are there other ways to remove it premanently? It has such a long tap root when I try to dif it up.
Expert Answer: It does look like Carolina Horsenettle (Solanum carolinense), though flowers (or if it stuck around long enough, \
fruits) would help to confirm the ID. It is native, but considered a weed in garden and agricultural settings. \ 
Either systemic herbicide to kill the roots or vigilant physical removal would be needed to eradicate it. If you \
wish to avoid herbicide, then dig up (or cut down) what you can, and remove any regrowth as quickly as it \
appears. Eventually, this will starve the roots of stored energy, and the plant(s) will stop regrowing. How long \
this process takes is hard to predict, but it might be several months at least if the plant(s) is well- \
established or mature. Even herbicide might take more than one application to be successful. Miri

Model Response: 
{'image_description': 'has long taproot', 'management_instructions': 'Either systemic herbicide to kill the roots \
or vigilant physical removal would be needed to eradicate it. If you wish to avoid herbicide, then dig up (or cut \
down) what you can, and remove any regrowth as quickly as it appears. Eventually, this will starve the roots of \
stored energy, and the plant(s) will stop regrowing. ', 'miscellaneous_facts': ['Carolina Horsenettle is native \
but considered a weed in garden and agricultural settings']}
</Example2>
"""

        prompt = prefix + "Here are some examples:" + example + "\n\n" + f"Follow the instructions and learn from the examples above to extract the content of following Q&A.\n\nTitle: {item['title']}\nUser Question: {item['question']}\nExpert Answer: {item['answer']}\n\nModel Response: "
       
        return {"prompt": prompt}
    
    # Prompts for Insect and Pest Management
    def get_prompt_ipm(self, item):
        prefix = """You are an assistant whose job it is to extract categories of information from an insect and pest management Q&A. \
These categories are "image description", "management instructions", and "miscellaneous facts". \
Make sure that "image descriptions" describe the visual qualities that can be referenced in the hypothetical image. \
If there is no mention of an image description in the conversation, output 'none'. \
Ensure that management instructions describe the recommended methods in the answer. \
Ensure that miscellaneous facts are independent of the exact situation and are self-contained atomic facts, meaning \
that each fact is a single coherent entity.

Format the response as a JSON. If no information for a specific category is present, output "none".
"""

        example = """
<Example1>
Title: Household flying insect identification
User Question: Can you please identify this insect? A number of these showed up in my kitchen a few days ago. I have killed about 10 of them, but they continue to emerge. I’d like to know where they most likely come from and how I can get rid of them.
Expert Answer: This looks like a Sunflower Seed Maggot Fly (Neotephritis finalis), a native insect widespread in North America. The larvae feed in immature seeds of many plants in the aster family (sunflowers being one of them). If you have raw sunflower seeds that weren't heat-treated or otherwise processed to kill insects, or are saving native plant seeds for propagation, perhaps that's where the population is coming from. These flies are not a typical household pest, so should not be a recurring problem if you can find the seeds they are breeding in and either remove them or seal their container.

Model Response:
{
  "image_description": "none",
  "management_instructions": "Locate the source of the infestation by checking for raw, untreated sunflower seeds or native plant seeds that may be harboring the larvae. Remove the infested seeds or seal their container to prevent further breeding.",
  "miscellaneous_facts": [
    "The Sunflower Seed Maggot Fly (Neotephritis finalis) is a native insect widespread in North America.",
    "The Sunflower Seed Maggot Fly's larvae feed in immature seeds of various plants in the aster family, including sunflowers.",
    "The Sunflower Seed Maggot Fly is not considered a typical household pest and should not reoccur if the breeding source is controlled."
  ]
}
<Example1/>

<Example2>
Title: Cherry Laurel
User Question: I have a stand of cherry laurels and one of the bushes has a white covering all over its bark. There are some dead branches on this particular bush. Can you tell me what it is and whether/how to treat it?
Expert Answer: That is called White Prunicola Scale and it is a common pest of Cherry Laurel. It is advisable to prune out the dead and heavily infected branches. Throughout the year, you can treat with an insecticidal soap or horticultural oil during the crawler emergence periods to help smother the nymphs. Once they take on the adult form and grow a protective coat, it becomes difficult to treat them, so manual removal such as scrubbing them off the branches or further pruning may be necessary. Let us know if you have further questions with treatment.

Model Response:
{
  "image description": "One of the bushes has a white covering all over its bark. There are some dead branches on this particular bush.",
  "management instructions": "Prune out dead and heavily infected branches to remove significant sources of infestation. Apply an insecticidal soap or horticultural oil during the crawler emergence period to target the vulnerable nymphs. If mature scales remain, manual removal, such as scrubbing them off the branches, may be necessary.",
  "miscellaneous facts": [
    "White Prunicola Scale is a common pest on cherry laurel.",
    "White Prunicola Scale is more susceptible to treatments during its crawler (nymph) stage.",
    "Once the White Prunicola Scale mature and develop a protective coating, they become much more difficult to control."
  ]
}
<Example2/>
"""

        prompt = prefix + "Here are some examples:" + example + "\n\n" + f"Follow the instructions and learn from the examples above to extract the content of following Q&A.\n\nTitle: {item['title']}\nUser Question: {item['question']}\nExpert Answer: {item['answer']}\n\nModel Response: "

        return {"prompt": prompt}

    # Plant Disease Management
    def get_prompt_pdm(self, item):
        prefix = """You are an assistant whose job it is to extract categories of information from a plant disease management Q&A. \
These categories are "symptom description", "management instructions", and "miscellaneous facts". \
Make sure that "symptom description" describes the symptoms of the disease. \
If there is no mention of disease symptoms in the conversation, output 'none'; do not speculate about symptom information. \
Ensure that management instructions describe the recommended methods in the answer. \
Ensure that miscellaneous facts are independent of the exact situation and are self-contained atomic facts, meaning \
that each fact is a single coherent entity.

Format the response as a JSON. If no information for a specific category is present, output "none".
"""

        example = """
<Example1>
Title: Veggie plant disease, affect on soil and compost for next year
Question: Hello! I am growing butternut squash (among other things) that have white spots and dying leaves. I also have tomatoes that have brown spots on yellow leaves (potentially early blight). Can I use this soil next year? It's in raised beds, and I really don't want to replace all of that soil. I also don't have too much space to rotate where I put things. I have tomatoes in different spots all over the garden beds. Additionally, before I realized about the white spots on the squash, I placed some leaves in my compost bin. It's a self-contained bin that I turn about once a week. Do I need to throw away the compost? What if I wait until next year to use it?
Answer: Early blight and powdery mildew can overwinter in the soil and on plant debris. Composting does not eliminate these pathogens. Ideally, crop rotation would be recommended. I would not use the compost containing the infected squash leaves in the garden. Early blight and powdery mildew are host specific, so the compost can be used around ornamental plants. If crop rotation and soil replacement are not an option, consider treating aggressively with fungicides at 7-day intervals next season after the crops are transplanted or emerged from the soil. Both diseases can migrate into the site on windborne spores, so even with crop rotation, soil replacement, and alternative compost uses, you could still see infection. Fungicide use is critical. If you wish to stay organic, consider organically approved fixed copper fungicides or biofungicides like Serenade, Regalia, or Double Nickel 55.

Model Response:
{
  "symptom_description": "Butternut squash show white spots and dying leaves; tomatoes exhibit brown spots on yellow leaves.",
  "management_instructions": "Do not use the compost containing infected squash leaves in the garden; ideally use crop rotation and soil replacement; if these options are not possible, treat aggressively with fungicides at 7-day intervals after transplanting or emergence; fungicide use is critical; for organic management, consider organically approved fixed copper fungicides or biofungicides such as Serenade, Regalia, or Double Nickel 55.",
  "miscellaneous_facts": [
    "Early blight and powdery mildew can overwinter in the soil and on plant debris.",
    "Composting does not eliminate early blight and powdery mildew.",
    "Early blight and powdery mildew are host specific.",
    "Early blight and powdery mildew can migrate into the site on windborne spores."
  ]
}
<Example1/>

<Example2>
Title: tomatos got problems
Question: Could you please tell me what's going wrong with my tomato plants? They are in big lots with lots of sun. Why are the leaves covered with brown spots and turning brown?
Answer: Why are your tomato leaves turning brown? It doesn't look like the brown patches are covering too many of the leaves. It could be a reaction to the cold weather that we had a week or two ago. If so, the problem will clear up once the summer gets underway. Remove the affected leaves and provide enough sun and water for the plants. If it's not due to the weather, then it could be the beginning of Blight - this is a fungal disease, characterized by spots on lower leaves and stems that appear water-soaked. Avoid overhead watering, and remove diseased leaves.

Model Response:
{
  "symptom description": "Tomato leaves are covered with brown spots and are turning brown. In the case of blight, lower leaves and stems may exhibit water-soaked spots.",
  "management instructions": "If the issue is due to cold weather, remove the affected leaves and ensure the plants receive adequate sun and water. If the symptoms indicate the onset of blight, characterized by water-soaked spots on lower leaves and stems, avoid overhead watering and remove diseased leaves.",
  "miscellaneous facts": [
    "Cold weather can cause tomato leaves to turn brown, with symptoms potentially resolving as the weather warms.",
    "Blight is a fungal disease characterized by water-soaked spots on lower leaves and stems."
  ]
}
<Example2/>
"""

        prompt = prefix + "Here are some examples:" + example + "\n\n" + f"Follow the instructions and learn from the examples above to extract the content of following Q&A.\n\nTitle: {item['title']}\nUser Question: {item['question']}\nExpert Answer: {item['answer']}\n\nModel Response: "

        return {"prompt": prompt}

    # Plant Care and Gardening Guidance
    def get_prompt_pcgg(self, item):
        prefix = """You are an assistant whose job it is to extract categories of information from a plant disease management Q&A. \
These categories are "image description", "management instructions", and "miscellaneous facts". \
Make sure that "image description" describes the symptoms of the disease. \
If there is no mention of an image description in the conversation, output 'none'. \
Ensure that management instructions describe the recommended methods in the answer. \
Ensure that miscellaneous facts are independent of the exact situation and are self-contained atomic facts, meaning \
that each fact is a single coherent entity.

Format the response as a JSON. If no information for a specific category is present, output "none"."""

        example = """
<Example1>
Title: Holly is dying
User Question: Hi, we have 3 holly trees in our backyard near the pool, and one of them has started dying very quickly (over the past 2-3 weeks). I initially noticed the leaves turning yellow, and some googling said it could be due to overwatering and iron deficiency(?). So I checked and there was a leaking valve from our saltwater pool right next to the tree and the ground was saturated. I turned off the valve and the leak stopped, but the tree continued to get significantly worse. Last week I purchased some soil acidifier and iron supplement that I poured around the roots and additional iron supplement that I sprayed on the tree and the ground, but it seems to still be dying? I really don't want to lose this tree so any help you can provide would be greatly appreciated. Thanks!
Expert Answer: Thanks for the submission. Unfortunately, salt damage is severe and most times (depending on tolerance) will kill a plant. Unfortunately, holly shrubs do not tolerate salt. The only way to remove salt from the root zone of the plant is to use gypsum and either water or rainwater. This will leach the salt away from the root zone of the holly. However, the next big issue will be overwatering due to the time of the year because not enough heat will not dry the soil out quickly enough.

Model Response:
{
  "image_description": "none",
  "management_instructions": "Apply gypsum along with water or rainwater to leach salt from the root zone. Monitor for overwatering issues due to cooler temperatures that slow soil drying.",
  "miscellaneous_facts": [
    "Holly shrubs are sensitive to salt and typically do not tolerate it.",
    "Salt damage is severe and can kill a plant.",
    "Gypsum can help remove salt from the root zone when used with water or rainwater.",
    "Cooler weather can contribute to overwatering issues because the soil does not dry quickly."
  ]
}
<Example1/>

<Example2>
Title: Tree Damage from Freeze
User Question: We have several cedar trees that seem to have browned seemingly overnight. They did well during the snow storm, but the freeze over the last couple of weeks really seems to have damaged them. I have read that freeze damage shows itself over a little time, but this has happened in a matter of days. What would be the best thing to try and save them (if possible)? I also noticed that almost all of these types in our neighborhood has similar damage, so it might be a good subject for an upcoming episode.
Expert Answer: Unfortunately this is a wait and see situation. This was an event that we haven't experienced for a very long time, if ever, and the full effects of the cold on plants won't be realized until the spring as plants break dormancy and begin growing again. Generally, needled evergreens like the cedar would not recover well when all the needles are killed, particularly true for pines, but there have been cases where cedars (those in the Cedrus genus) recover and send out new growth in the spring. One of the key factors will be if the buds that will provide the new growth for this year were damaged. Right now all we can do is wait. As the weather begins to warm and new growth (hopefully) appears this spring, keeping the plants properly watered and fertilized to encourage vigorous growth and avoid further stress will be important. Hopefully this helps.

Model Response:
{
  "image description": "Cedar trees that appear browned overnight.",
  "management instructions": "Wait until spring to assess recovery. In the meantime, keep the trees properly watered and fertilized to encourage new growth and reduce further stress.",
  "miscellaneous facts": [
    "Freeze damage in plants may not be fully evident until spring when they break dormancy.",
    "Needled evergreens generally do not recover well if all their needles are killed.",
    "Some cedars, particularly those in the Cedrus genus, can recover and produce new growth if the buds are undamaged."
  ]
}
<Example2/>
"""
        prompt = prefix + "Here are some examples:" + example + "\n\n" + f"Follow the instructions and learn from the examples above to extract the content of following Q&A.\n\nTitle: {item['title']}\nUser Question: {item['question']}\nExpert Answer: {item['answer']}\n\nModel Response: "  
        return {"prompt": prompt}

    # Plant Identification
    def get_prompt_pi(self, item):
        prefix = """You are an assistant whose job it is to extract categories of information from a plant identification Q&A. \
These categories are "image description", and "miscellaneous facts". \
Make sure that "image description" describes the symptoms of the disease. \
If there is no mention of an image description in the conversation, output 'none'. \
Ensure that miscellaneous facts are independent of the exact situation and are self-contained atomic facts, meaning \
that each fact is a single coherent entity.

Format the response as a JSON. If no information for a specific category is present, output "none"."""

        example = """\
<Example1>
Title: What kind of tree is this? Is this edible or is it toxic to humans and dogs ?
User Question: It looks like a berry but color is between a white and grape. Tree is about 8 feet tall. Is it edible or toxic to humans and dogs?
Expert Answer: Morus alba, White mulberry tree, is a nonnative tree. All parts of the tree contain latex which is toxic if consumed. The completely ripe berries can be consumed by humans. There is no information on toxicity to dogs, so it is probably better to play it safe and refrain from consuming them. Birds can eat them without ill effect.

Model Response:
{
  "image_description": "none",
  "miscellaneous_facts": [
    "Morus alba, known as the white mulberry tree, is a nonnative species.",
    "All parts of the morus alba tree contain latex which is toxic if ingested.",
    "Only completely ripe morus alba berries are edible for humans.",
    "There is no definitive information on the toxicity of the morus alba berries to dogs, so caution is advised.",
    "Birds can consume the morus alba berries without ill effect."
  ]
}
<Example1/>

<Example2>
Title: Tree with bright yellow berries in autumn
User Question: Please help me identify this lovely tree with bright yellow berries. Saw it in several places in late October. No leaves on tree, but from the ground waste the leaves look elm-like. Attached are pics of the bark, berries, & tree from a distance.
Expert Answer: This is a yellow-fruited flowering crab. Not as many of these on the market as the red ones but they are definitely beautiful! I also enjoyed them this fall. Must have been a good year for abundant fruits.

Model Response:
{
  "image_description": "none",
  "miscellaneous_facts": [
    "The tree is identified as a yellow-fruited flowering crab.",
    "Yellow-fruited flowering crabs are less common than red-fruited varieties.",
    "Yellow-fruited flowering crabs can produce abundant fruit in favorable years.",
    "Yellow-fruited flowering crabs is noted for its attractive appearance, especially with bright yellow berries in autumn."
  ]
}
<Example2/>
"""
        prompt = prefix + "Here are some examples:" + example + "\n\n" + f"Follow the instructions and learn from the examples above to extract the content of following Q&A.\n\nTitle: {item['title']}\nUser Question: {item['question']}\nExpert Answer: {item['answer']}\n\nModel Response: "
        
        return {"prompt": prompt}
    
    # Insect and Pest Identification
    def get_prompt_ipi(self, item):
        prefix = """You are an assistant whose job it is to extract categories of information from a insect and pest identification Q&A. \
These categories are "image description", and "miscellaneous facts". \
Make sure that "image description" describes the symptoms of the disease. \
If there is no mention of an image description in the conversation, output 'none'. \
Ensure that miscellaneous facts are independent of the exact situation and are self-contained atomic facts, meaning \
that each fact is a single coherent entity.

Format the response as a JSON. If no information for a specific category is present, output "none"."""

        example = """\
<Example1>
Title: What kind of big is this?
User Question: These bugs are all over lately and trying to find out what they are and why I have just started seeing them?
Expert Answer: Hello, The insect in your photo is a stag beetle. The large mandibles are used for breakdown wood. They are decomposers and considered beneficial insects. There must be a stump or dead wood as a food source for them nearby. They are also attracted to lights at night. Regards.

Model Response:
{
  "image_description": "none",
  "miscellaneous_facts": [
    "The insect in the photo is a stag beetle.",
    "Stag beetles have large mandibles that are used to break down wood.",
    "Stag beetles are decomposers and are considered beneficial insects.",
    "A nearby stump or dead wood is likely providing a food source for Stag beetles.",
    "Stag beetles are attracted to lights at night."
  ]
}
<Example1/>

<Example2>
Title: what kind of roach
User Question: I found this crawling around near my kitchen door this evening. When I tried to catch it, it hopped and ran. I have found two or three in the house before a couple of years ago, again near the kitchen door and once near a sunroom which has a sliding door. All the times I found them in the fall, usually in the evening, but one during the day. Always this size too. I have not seen any adult roaches in or outside the house in the 7 years we've lived here. We find a lot of beetles and other strange bugs in the house often but not roaches. I'm pretty sure this is a roach even though I've never seen one this small. Thanks for your help.
Expert Answer: I don't think that is a cockroach. Instead, I think it is a young camel cricket. The arched body form and the enlarged hind legs are not what one would see on a cockroach. Plus, your observations of its activity, being out and around in the sunroom during the day, is more consistent with a cricket than a cockroach, which often hides during the day.

{
  "image_description": "none",
  "miscellaneous_facts": [
    "The insect is likely a young camel cricket rather than a cockroach.",
    "Camel crickets have an arched body form and enlarged hind legs, which differ from those of cockroaches.",
    "The observed behavior of being active during the day is more consistent with a cricket than with a cockroach, which typically hides during the day."
  ]
}
<Example2/>
"""
        

        prompt = prefix + "Here are some examples:" + example + "\n\n" + f"Follow the instructions and learn from the examples above to extract the content of following Q&A.\n\nTitle: {item['title']}\nUser Question: {item['question']}\nExpert Answer: {item['answer']}\n\nModel Response: "
        return {"prompt": prompt}
    
    # Plant Disease Indentification
    def get_prompt_pdi(self, item):
        prefix = """You are an assistant whose job it is to extract categories of information from a plant disease identification Q&A. \
These categories are "symptom description", and "miscellaneous facts". \
Make sure that "symptom description" describes the symptoms of the disease. \
If there is no mention of disease symptoms in the conversation, output 'none'; do not speculate about symptom information. \
Ensure that miscellaneous facts are independent of the exact situation and are self-contained atomic facts, meaning \
that each fact is a single coherent entity.

Format the response as a JSON. If no information for a specific category is present, output "none".
"""

        example = """\
<Example1>
Title: could this be frost damage?
User Question: We have several peonies showing this damage/symptom. Not all peonies show it. The damage appears on top of the leaf. The damage has extended to whitish dead areas along the vein today.
Expert Answer: Yes, it could be frost/cold damage; we do not see evidence of pest or disease issues. Nothing needs to be done at this point and it is highly unlikely we will see such temperatures again this spring, but if new damage appears, please send us more photos for diagnosis.

Model Response:
{
  "symptom_description": "Peony leaves show damage on the top surface with whitish, dead areas developing along the veins.",
  "miscellaneous_facts": [
    "Frost or cold damage results from exposure to low temperatures rather than pests or pathogens.",
    "Frost damage may not affect all plants uniformly and often does not require intervention if temperatures improve."
  ]
}
<Example1/>

<Example2>
Title: Brown spots on plants
User Question: My plants have rusty brown spots on my new plants.
Expert Answer: What is affecting your plants is not rust, nor is it insect damage. It is a fungal-type disorder which comes from erratic weather (wet, then too hot, etc.). Similar damage can be caused by re-using the soil.

Model Response:
{
  "symptom_description": "New plants exhibit rusty brown spots on their foliage.",
  "miscellaneous_facts": [
    "The disorder is fungal in nature, not caused by rust or insect damage.",
    "Erratic weather conditions, such as periods of wetness followed by excessive heat, can trigger this fungal disorder.",
    "Re-using soil can lead to similar fungal disorder on plants."
  ]
}
<Example2/>
"""

        prompt = prefix + "Here are some examples:" + example + "\n\n" + f"Follow the instructions and learn from the examples above to extract the content of following Q&A.\n\nTitle: {item['title']}\nUser Question: {item['question']}\nExpert Answer: {item['answer']}\n\nModel Response: "
        return {"prompt": prompt}

    def process_item(self, args):
        item, model_name, output_file, lock = args

        if self.model_name == "gpt-4o" or self.model_name == "gpt-4o-mini":
            client = GPT4O(model_name=model_name, messages=[])
        else:
            raise ValueError(f"Model '{self.model_name}' not supported.")
      
        try:
            if item.get('category') == "Weeds/Invasive Plants Management":
                prompt = self.get_prompt_wipm(item)
                response = client.chat(prompt=prompt["prompt"], response_format=TAGS)
                item["tags"] = {"image_description": response.image_description, "management_instructions": response.management_instructions, "miscellaneous_facts": response.miscellaneous_facts}
            elif item.get('category') == "Insect and Pest Management":
                prompt = self.get_prompt_ipm(item)
                response = client.chat(prompt=prompt["prompt"], response_format=TAGS)
                item["tags"] = {"image_description": response.image_description, "management_instructions": response.management_instructions, "miscellaneous_facts": response.miscellaneous_facts}
            elif item.get('category') == "Plant Care and Gardening Guidance":
                prompt = self.get_prompt_pcgg(item)
                response = client.chat(prompt=prompt["prompt"], response_format=TAGS)
                item["tags"] = {"image_description": response.image_description, "management_instructions": response.management_instructions, "miscellaneous_facts": response.miscellaneous_facts}
            elif item.get('category') == "Plant Disease Management":
                prompt = self.get_prompt_pdm(item)
                response = client.chat(prompt=prompt["prompt"], response_format=TAGS2)
                item["tags"] = {"symptom_description": response.symptom_description, "management_instructions": response.management_instructions, "miscellaneous_facts": response.miscellaneous_facts}
            elif item.get('category') == "Plant Identification":
                prompt = self.get_prompt_pi(item)
                response = client.chat(prompt=prompt["prompt"], response_format=TAGS3)
                item["tags"] = {"image_description": response.image_description, "miscellaneous_facts": response.miscellaneous_facts}
            elif item.get('category') == "Insect and Pest Identification":
                prompt = self.get_prompt_ipi(item)
                response = client.chat(prompt=prompt["prompt"], response_format=TAGS3)
                item["tags"] = {"image_description": response.image_description, "miscellaneous_facts": response.miscellaneous_facts}
            elif item.get('category') == "Plant Disease Identification":
                prompt = self.get_prompt_pdi(item)
                response = client.chat(prompt=prompt["prompt"], response_format=TAGS4)
                item["tags"] = {"symptom_description": response.symptom_description, "miscellaneous_facts": response.miscellaneous_facts}
            item["info"] = client.info()
            item["history"] = client.get_history()
        except Exception as e:
            print(f"Error processing item {item.get('id', 'unknown')}: {e}")
            item["tags"] = None
            
        with lock:
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        return item.get('id')

    def extract(self):
        with open(self.raw_data_file, "r", encoding='utf-8') as f:
            data = json.load(f)

        processed_ids = set()
        if os.path.exists(self.output_file):
            with open(self.output_file, "r", encoding='utf-8') as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        if "tags" in item and item["tags"] is not None:
                            processed_ids.add(item['id'])
                    except json.JSONDecodeError:
                        continue
                    
        items_to_process = [item for item in data if item.get('id') not in processed_ids]
        print(f"Processing {len(items_to_process)} items.")
        
        if items_to_process:
            manager = multiprocessing.Manager()
            lock = manager.Lock()
            pool = multiprocessing.Pool(processes=self.num_processes)
            args_list = [(item, self.model_name, self.output_file, lock) for item in items_to_process]
            for _ in tqdm(pool.imap_unordered(self.process_item, args_list), total=len(args_list), desc="Processing items"):
                pass
            pool.close()
            pool.join()
        
        print("Processing completed.")
        self.cleanup_output(len(data))

    def cleanup_output(self, data_length):
        valid_items = []
        
        with open(self.output_file, "r", encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    if "tags" in item and item["tags"] is not None:
                        valid_items.append(item)
                except json.JSONDecodeError:
                    continue

        with open(self.output_file, "w", encoding='utf-8') as f:
            for item in valid_items:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        print(f"Total successful items: {len(valid_items)}. \n Remaining items to process: {data_length - len(valid_items)}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract facts from agriculture questions using LLMs model.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output JSONL file.")
    parser.add_argument("--model_name", type=str, default="gpt-4o", help="Model name to use.")
    parser.add_argument("--num_processes", type=int, default=os.cpu_count(), help="Number of processes to use.")
    args = parser.parse_args()

    extractor = Fact_Extractor(args.input_file, args.output_file, args.model_name, args.num_processes)
    extractor.extract()
