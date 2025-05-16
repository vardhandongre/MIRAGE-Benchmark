def detect_speaker(responder, idx):
    responder = responder.lower()
    if "question asker" in responder:
        return "user"
    return "expert"

def truncate_dialog(dialog, last_user_idx, max_user_turns=3):
    context = []
    user_count = 0
    for turn in reversed(dialog[:last_user_idx]):
        context.insert(0, turn)
        if turn["speaker"] == "user":
            user_count += 1
        if user_count >= max_user_turns:
            break
    return context

def get_image_captions(entry, caption_lookup):
    images = entry.get("attachments", [])
    return [caption_lookup.get(img_id, f"Image {img_id}") for img_id in images]