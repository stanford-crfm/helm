from typing import List

from helm.benchmark.scenarios.scenario import Scenario, Instance, Input, TEST_SPLIT


class DailyDallEScenario(Scenario):
    """
    DALL-E 2 prompts from Chad Nelson's Instagram: https://www.instagram.com/dailydall.e
    Chad Nelson was a featured artist on OpenAI's blogpost:
    https://openai.com/blog/dall-e-2-extending-creativity.
    """

    PROMPTS: List[str] = [
        "a lone hairy monster is struggling to walk in a snow storm, a rusty metal sign points to HOME, backlit",
        "a Ukrainian soldier in winter, rack focus, close-up, portrait photography",
        "close-up of a snow leopard in the snow hunting, rack focus, nature photography",
        "a cute furry monster dressed as a pirate for Halloween goes trick-or-treating in a misty forest",
        "a cargo hangar interior from the TV show Space 1999, dramatic lighting",
        "a SPACE: 1999 designed orange and white interplanetary transport with rocket engines, radar "
        "and landing gear on Mars during a sand storm",
        "a delicious cocktail on a wooden table next to the beach, rack focus, sunny day, travel photography",
        "sand dunes at sunrise, dramatic light, strong contrasting shadows, nature photography, "
        "Death Valley National Park",
        "a old retro van built to TIME TRAVEL",
        "a old retro van built to chase UFOs",
        "an old Sprinter style camper van from the 1960s that is built to chase dreams",
        "a geometric painting of circles and shapes for an urban building, mural art",
        "a vintage retro rocket blasts off towards the moon, silk screen poster style",
        "a cute furry bear with black and white stripes sits and enjoys coffee, close-up with selective focus",
        "a group of furry black and white striped monsters scream in excitement at a concert, close-up "
        "with selected focus",
        "a vintage Land Rover Defender drives within a dramatic vista in Monument Valley, cinematic sky and light",
        "a little girl at the entrance of a bottomless hole that is filled with light, backlit, looking down "
        "from above",
        "a girl stands frozen in shock as she looks at a bright illuminated light, within a dark misty forest",
        "an old RV illuminated from inside is parked in the misty woods at night, wide shot",
        "a group of happy red monsters celebrate as confetti falls from the ceiling",
        "a tricked-out red RV built to hunt UFOs, digital art",
        "a robot sits at a table about to eat some cereal",
        "a skull of a robot alien displayed in a museum",
        "an extreme close-up of a man taking pictures with an old vintage hand-held camera, film noir style",
        "a alien astronaut in the cockpit of a retro spaceship, 1950s scifi style",
        "the glow of a burning fire within a futuristic refinery",
        "a cute yellow furry monster is in panic from a fire in the misty forest",
        "an astronaut looks at a retro rocket ship from inside a dark hanger",
        "a cute yellow furry monster walks into a misty forest",
        "the patio of a modern home made of glass wood and steel in Joshua Tree",
        "a furry red monster questioning life choices",
        "a retro rocket whooshing to the moon, silk screen poster style",
        "a lone monster walks in a forest during a misty sunrise, pulp illustration style",
        "comic book style illustration of a UFO abduction",
        "a happy pirate plays golf on the beach, pixel art style",
        "a friendly robot meets a kitten",
        "schematic posters for 1960s space craft, silk screen print style",
        "a happy furry white caterpillar marvels at fireflies in a misty forest",
        "an alien robot spider emerges from a desert sandstorm, dramatic light",
        "a cybernetic solider from the future",
        "a modern robot performs data entry on a computer",
        "a red furry spider hangs from a tree branch in a misty forest",
        "a cute furry monster relaxes in the tree branches within a misty forest",
        "a big white furry monster shakes it’s hips and raises it’s arms disco dancing, dramatic lighting",
        "a father and son sit in the window of a futuristic space station overlooking other planets, backlit",
        "a glamorous woman in 1970s disco fashion, backlit over white background, high-end fashion photography",
        "a massive rusty robot and a cute furry forest critter explore the misty forest",
        "a small boy discovers a large mechanical robot with green eyes in the misty forest",
        "a yellow striped monster in panic while working on a laptop",
        "a cute happy dinosaur celebrating a birthday in the desert",
        "a baby T-Rex is excited celebrating a birthday with confetti and balloons",
        "a security robot inside an empty London Underground, dramatic lighting, looking up from the ground, "
        "pinhole photography",
        "a NASA JPL inspired large cargo communications transport vehicle from the future, on deserted salt flats",
        "a little red furry monster is excited jumping over a mound in a misty forest",
        "New Zealand Mt Cook with a river leading into a beautiful meadow in fall, low clouds, sunrise",
        "a hairy blue monster wakes up in complete panic in bed, alarm clock on a bedside table",
        "a big blue furry monster takes a nap in the misty forest",
        "a SciFi robotic brain connected to computers and an retro TV showing data, dramatic lighting",
        "a NASA design inspired large cargo personnel planetary transport vehicle, on a flat barren desert planet",
        "a wise old hairy critter wanders alone through the desert on two feet",
        "a yellow furry Dad monster lovingly hugs his two happy little yellow furry kid monsters in a misty forest",
        "a 1960s-era retro device for displaying recipes set on a kitchen counter, single dramatic light source",
        "a 1960s-era handheld communication device on an old metal table",
        "an old retro phone with a digital display and push-buttons, single light source",
        "a scifi retro handheld walkie-talkie on a metal table, single light source through blinds",
        "a scifi retro portable brain scanning device, single light source",
        "a retro scifi medical scanner, single light source",
        "a retro scifi handheld communications device, on a grated metal table, single light source",
        "a retro scifi handheld scanning device, single light source",
        "a close-up of a painted metal tiger figurine on an old metal table lit with a single directional light, "
        "high contrast",
        "a pewter retro rocket on a brushed metal table with dramatic contrasting light",
        "a happy monster relaxing on a pool floaty holding a refreshing tiki drink",
        "a white hairy monster family smiles for a selfie, camera looking up, in New York City",
        "a black furry monster zooms high above New York City, close up with motion blur",
        "a giant white furry monster stomps into a city, camera looking up from street view",
        "a cute green furry monster waves goodbye to a friend in a misty forest",
        "a curious blue striped furry monster climbs a tree, surprised by a bee within a misty forest",
        "a cute little yellow monster with flower horns smiles within a misty forest",
        "a clever furry monster joyfully rises from the moss within a misty forest",
        "a hairy red spider with big eyes hangs from a tree branch within a misty forest",
        "an angry green hairy monster in a misty forest",
        "two furry monsters explore a cemetery in a misty forest for Memorial Day",
        "a happy blue monster with horns hides behind a log in a misty forest",
        "a short furry monster with black fur walks out of a misty forest, silhouette",
        "a short furry monster living in a misty forest standing on a tree branch",
        "a lone man walks down the rainy city backstreets illuminated by orange and cyan lights",
        "Macro photography of a vintage toy robot caught in a snow storm",
        "Product photography for a retro sci-fi laser scanning device",
        "a short furry yellow monster with a buck tooth explores a misty forest",
        "a giant robot spider walks into a futuristic city",
        "an ice cream monster",
        "an astronaut sits within a futurist cockpit overlooking Jupiter",
        "a red furry monster looks in wonder at a burning candle",
    ]

    name = "daily_dalle"
    description = (
        "DALL-E 2 prompts from [Chad Nelson's Instagram](https://www.instagram.com/dailydall.e/)"
        "Chad Nelson was a featured artist on [OpenAI's blogpost]"
        "(https://openai.com/blog/dall-e-2-extending-creativity)."
    )
    tags = ["text-to-image", "originality"]

    def get_instances(self, _) -> List[Instance]:
        return [Instance(Input(text=prompt), references=[], split=TEST_SPLIT) for prompt in self.PROMPTS]
