def get_instruction():
    return ["""Analyze the video frames to identify up to five significant objects. Focus on clear, discernible objects integral to the video's context. Provide detailed descriptions including color, shape, actions, styles, movements, and distinct features. Ensure precision and brevity for clarity and reference.

Tasks:

1. Provide a concise holistic video description, including main objects, actions, and events.
2. Catalog up to five significant objects within the video.

Format:

Video Description:
[Example] Drone view of waves crashing against rugged cliffs along Big Sur's beach. The blue waters create white-tipped waves, illuminated by the setting sun. A lighthouse sits on a small island, and green shrubbery covers the cliffs. The steep drop from the road to the beach highlights the coast's raw beauty.

Object Descriptions:
1. Woman: Stylish woman in a black leather jacket, long red dress, and black boots. She carries a black purse, wears sunglasses and red lipstick, and walks confidently.
2. Object 2: [Description]
... (up to 5 objects)

Ensure descriptions are concise, thorough, and highlight relationships between objects if relevant."""]
    
    
#     return ["""Analyze the video frames, focusing on identifying up to five significant objects. Provide a comprehensive video description and catalog the key objects, ensuring your descriptions are detailed and structured as follows:

# Video Description: Offer a holistic view of the video's context, highlighting key actions and events. Avoid repetitive or frame-level descriptions.
# Object Catalog: List and describe up to five significant objects, detailing their appearance, relationships, and significance within the video. Your descriptions should be precise and thorough, covering attributes such as color, shape, style, and movement.

# Example Formats:

# Video Description: "Drone view of waves crashing against Big Sur's cliffs, illuminating the rocky shore. A lighthouse island and green cliffs create a rugged coastline view."
# Object Catalog: "1. Woman: A stylish woman in a black leather jacket, red dress, and black boots, confidently walking with a black purse." (Continue for others)
# Ensure clarity and specificity to aid in comprehension.
# """,

# """Analyze the video frames, identifying the most important object. Provide a description following this structure:

# Focus on clear, discernible objects integral to the video's context, capturing key attributes such as color, shape, actions, styles, movements, and distinct features. Aim for precision and succinctness to aid comprehension.
# Object Description: Offer a detailed description of the object's appearance and significance within the video, ensuring clarity and specificity.
# Example Format:
# "Woman: A stylish woman in a black leather jacket, red dress, and black boots, carrying a black purse. She wears sunglasses and red lipstick, walking confidently and casually."

# Ensure your description is concise, thorough, and aids in reconstructing the object in your mind.
# """,

# """Assuming the frame size is normalized to the range 0-1000, carefully analyze the successive frames provided in the video. Expect the presence of "<TARGET_OBJ>" and its layouts within these frames. Ensure that your analysis adheres to the structure outlined below, prioritizing objects that are clear, discernible, and integral to the overall context of the video. 

# You need to generate layouts from a close-up camera view of the event. The layout of the object to be reconstructed in each frame is represented as a rectangle or square box, with the layout and size of the boxes as large as possible. The layout difference between two adjacent frames must be small, considering the small interval. You also need to generate a caption that best describes the image for each frame.

# Example Format for Listing frame index and layouts of the <TARGET_OBJ> as a format of [left, top, right, bottom]:

# Frame 1: [250, 250, 1000, 700],
# Frame 2: [250, 250, 1000, 700],
# ...

# Ensure your description is both concise and thorough, providing clear insight into the object's appearance and its significance within the video. Your descriptions should excel in clarity, detail, and specificity, ensuring you can reconstruct the objects in your mind according to your descriptions.
# """
# ] 
# """
# Assuming the frame size is normalized to the range 0-1000, carefully analyze the successive frames provided in the video. Describe "<TARGET_OBJ>" and identify its presence within these frames. Ensure that your analysis adheres to the structure outlined below, prioritizing objects that are clear, discernible, and integral to the overall context of the video. Your descriptions should be rich in detail, capturing key attributes such as color, shape, actions, styles, movements, and any distinct features. Strive for precision and succinctness to aid in comprehension and ease of reference.

# The object in each frame is represented as a rectangle or square box, with the layout and size of the boxes as large as possible. You need to generate layouts from a close-up camera view of the event. The layout difference between two adjacent frames must be small, considering the small interval. You also need to generate a caption that best describes the image for each frame.

# Example Format for Listing Identified Object Name, its Description, and layouts as a format of [left, top, right, bottom]:

# Woman: A stylish woman. She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. She wears sunglasses and red lipstick. She walks confidently and casually.
# Frame 1: {'<TARGET_OBJ>': [250, 250, 1000, 700]},
# Frame 2: {'<TARGET_OBJ>': [250, 250, 1000, 700]},
# ...

# Ensure your description is both concise and thorough, providing clear insight into the object's appearance and its significance within the video. Your descriptions should excel in clarity, detail, and specificity, ensuring you can reconstruct the objects in your mind according to your descriptions.
# """
    
    
    
    return ["""Carefully analyze the successive frames provided in the video, focusing on identifying and cataloging up to five significant objects within these frames. Ensure that your analysis adheres to the structure outlined below, prioritizing objects that are clear, discernible, and integral to the overall context of the video. Your descriptions should be rich in detail, capturing key attributes such as color, shape, actions, styles, movements, and any distinct features. Strive for precision and succinctness to aid in comprehension and ease of reference.

                    Please focus on two tasks (1) providing a short and well-organized holistic video description including main objects, actions, and events; (2) identifying and cataloging (up to five) significant objects within this video. 

                    Ensure that your analysis adheres to the structure outlined below, prioritizing objects that are clear, discernible, and integral to the overall context of the video. Your descriptions should be rich in detail, capturing key attributes such as color, shape, actions, styles, movements, and any distinct features. Strive for precision and succinctness to aid in comprehension and ease of reference.

                    Format for Video Description (Ensure to avoid repetitive descriptions and frame-level descriptions.):
                                
                    Video Description: Drone view of waves crashing against the rugged cliffs along Big Sur's garay point beach. The crashing blue waters create white-tipped waves, while the golden light of the setting sun illuminates the rocky shore. A small island with a lighthouse sits in the distance, and green shrubbery covers the cliff's edge. The steep drop from the road down to the beach is a dramatic feat, with the cliff's edges jutting out over the sea. This is a view that captures the raw beauty of the coast and the rugged landscape of the Pacific Coast Highway.
                                
                    Example Format for Listing Identified Object Names and their Descriptions:

                    1. Woman: A stylish woman. She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. She wears sunglasses and red lipstick. She walks confidently and casually. 
                    ... (Continue if there are other significant objects)
                                
                    Number each object uniquely, ensuring your descriptions are both concise and thorough, providing clear insight into each object's appearance and its significance within the video. Highlight any relationships between objects if they are part of a larger, significant item or scene. 
                    Your descriptions should excel in clarity, detail, and specificity, ensuring you can reconstruct the objects in your mind according to your descriptions. 
                    Do not provide explanations for the generated captions.""",
                    
                    """Carefully analyze the successive frames provided in the video, focusing on identifying the most important object within these frames. Ensure that your analysis adheres to the structure outlined below, prioritizing objects that are clear, discernible, and integral to the overall context of the video. Your descriptions should be rich in detail, capturing key attributes such as color, shape, actions, styles, movements, and any distinct features. Strive for precision and succinctness to aid in comprehension and ease of reference.

                    Please focus on the task: identifying the most significant object within this video. 

                    Ensure that your analysis adheres to the structure outlined below, prioritizing objects that are clear, discernible, and integral to the overall context of the video. Your descriptions should be rich in detail, capturing key attributes such as color, shape, actions, styles, movements, and any distinct features. Strive for precision and succinctness to aid in comprehension and ease of reference.

                    Example Format for Listing Identified Object Names and their Descriptions:

                    Woman: A stylish woman. She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. She wears sunglasses and red lipstick. She walks confidently and casually. 
                    
                    Ensuring your description is both concise and thorough, providing clear insight into the object's appearance and its significance within the video.                    
                    Your descriptions should excel in clarity, detail, and specificity, ensuring you can reconstruct the objects in your mind according to your descriptions. 
                    Do not provide explanations for the generated captions.""",
                    
                    """
                    Assuming the frame size is normalized to the range 0-1, carefully analyze the successive frames provided in the video. Describe "<TARGET_OBJ>" and identify its presence within these frames. Ensure that your analysis adheres to the structure outlined below, prioritizing objects that are clear, discernible, and integral to the overall context of the video. Your descriptions should be rich in detail, capturing key attributes such as color, shape, actions, styles, movements, and any distinct features. Strive for precision and succinctness to aid in comprehension and ease of reference.

                    The object in each frame is represented as a rectangle or square box, with the layout and size of the boxes as large as possible. You need to generate layouts from a close-up camera view of the event. The layout difference between two adjacent frames must be small, considering the small interval. You also need to generate a caption that best describes the image for each frame.

                    Example Format for Listing Identified Object Name, its Description, and layouts as a format of [left, top, right, bottom]:

                    Woman: A stylish woman. She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. She wears sunglasses and red lipstick. She walks confidently and casually.
                    Frame 1: [0.25, 0.25, 1.00, 0.70],
                    Frame 2: [0.25, 0.25, 1.00, 0.70],
                    ...

                    Ensure your description is both concise and thorough, providing clear insight into the object's appearance and its significance within the video. Your descriptions should excel in clarity, detail, and specificity, ensuring you can reconstruct the objects in your mind according to your descriptions.
                    """
                    ]