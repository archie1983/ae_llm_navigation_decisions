import pickle
import os.path

from time import time

from .ae_llm import LLMControl, LLMType
from .room_type import RoomType

##
# A class that is sort of a middle man between the LLM that we will use for
# classifying rooms and the data that is being provided.
##
class LLMDecisionMaker:
    def __init__(self):
        self.data_counter = 0
        self.false_cnt = 0
        self.true_cnt = 0

        self.stored_labels_loaded = False

        self.glc = LLMControl(LLMType.MISTRAL_4b)

    def __init__(self, llm_type):
        self.data_counter = 0
        self.false_cnt = 0
        self.true_cnt = 0

        self.stored_labels_loaded = False

        self.glc = LLMControl(llm_type)

    def classify_room_by_this_object_set(self, obj_set):
        # now we'll get the objects into a string separated by a space
        objs_in_room_as_string = ""
        for obj in obj_set:
            objs_in_room_as_string += obj + ", "

        objs_in_room_as_string = objs_in_room_as_string[:-2]

        self.glc.construct_classifier_question(objs_in_room_as_string)

        #t0 = time()
        ans, full_text = self.glc.get_answer()
        #print("llm predict time:", round(time()-t0, 3), "s")

        #print("\n" + str(ans) + " :: " + list(self.room_types.keys())[list(self.room_types.values()).index(ans)])
        #print("\n" + ans.name + " :: " + str(ans.value))

        return ans, full_text

    def classify_room_by_this_object_set_and_pic(self, obj_set = None, img_bytes = None):
        # now we'll get the objects into a string separated by a space
        if obj_set is not None:
            objs_in_room_as_string = ""
            for obj in obj_set:
                objs_in_room_as_string += obj + ", "

            objs_in_room_as_string = objs_in_room_as_string[:-2]

        if img_bytes is None:
            self.glc.construct_classifier_question_multi_modal_no_img(objs_in_room_as_string)
        else:
            if obj_set is None:
                self.glc.construct_classifier_question_multi_modal_img_only()
            else:
                self.glc.construct_classifier_question_multi_modal(objs_in_room_as_string)

        #t0 = time()
        ans, full_text = self.glc.get_answer(img_uri=None, img_bytes=img_bytes)

        ## Re-parse the full text for the answer, because for these multi-modal LLMs we ask to precede
        # the correct answer with a $ sign - because they're quite chatty and mention all sorts of rooms before
        # the final answer.
        if "$" in full_text:
            txt_to_parse = full_text[full_text.rfind("$"):]
            ans = RoomType.parse_llm_response(txt_to_parse)
        #print("llm predict time:", round(time()-t0, 3), "s")

        #print("\n" + str(ans) + " :: " + list(self.room_types.keys())[list(self.room_types.values()).index(ans)])
        #print("\n" + ans.name + " :: " + str(ans.value))

        return ans, full_text

    ##
    # Allows us asking the LLM where to find a given object
    ##
    def where_to_find_this(self, object_name):
        self.glc.construct_room_selector_question(object_name)
        ans, full_text = self.glc.get_answer()

        return ans, full_text

    def where_to_look_first(self, what_to_look_for, obj_list_to_explore):
        objs_to_look_near = ""
        for obj in obj_list_to_explore:
            objs_to_look_near += obj + ", "

        objs_to_look_near = objs_to_look_near[:-2]

        self.glc.construct_object_selector_question(what_to_look_for, objs_to_look_near)
        ans, full_answer_unmodified = self.glc.get_object_selector_answer(obj_list_to_explore)

        print("ANS: ", ans)
        return ans, full_answer_unmodified

if __name__ == "__main__":
    ldm = LLMDecisionMaker(LLMType.MINISTRAL_3_3b_instruct_nf4_bnb)
    rt_llm, llm_text = ldm.classify_room_by_this_object_set_and_pic(obj_set="Scales, bathtub, toothbrush", img_uri=None)
    print(rt_llm, llm_text)

#print(rc.get_next_data_item())
#print(rc.get_next_data_item())
#print(rc.get_next_data_item())
#print(rc.get_next_data_item())

#rc.predict("SinkBasin CounterTop SoapBar ToiletPaperHanger")
#rc.predict("SinkBasin Chair Egg Toaster Microwave CounterTop DiningTable StoveKnob Lettuce SaltShaker")
#rc.predict("SinkBasin Chair Egg Toaster Microwave CounterTop DiningTable StoveKnob Lettuce")
#rc.predict("Egg")
##rc.predict("SinkBasin CounterTop SoapBar ToiletPaperHanger ToiletPaper SprayBottle Floor GarbageCan Candle Plunger ScrubBrush Toilet Sink HandTowelHolder Faucet Mirror Cloth Towel Drawer SoapBottle ShowerHead HandTowel LightSwitch ShowerDoor TowelHolder ShowerGlass")
##rc.predict("Candle Plunger ScrubBrush Toilet Sink HandTowelHolder SoapBottle")
#rc.predict("Candle Plunger ScrubBrush Toilet")
#rc.predict("TV Sofa")
#rc.predict("ScrubBrush ToiletCandle Plunger")
