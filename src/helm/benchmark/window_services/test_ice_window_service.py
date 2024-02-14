import shutil
import tempfile
from typing import List

from helm.common.cache_backend_config import BlackHoleCacheBackendConfig
from .tokenizer_service import TokenizerService
from .window_service_factory import WindowServiceFactory
from .test_utils import get_tokenizer_service, TEST_PROMPT


class TestICEWindowService:
    # According to https://github.com/THUDM/icetk, token id [20100, 83823) are English tokens.
    TEST_TOKEN_IDS: List[int] = [
        20123,
        21490,
        20108,
        22581,
        20111,
        22430,
        48828,
        20019,
        21172,
        27993,
        20014,
        20107,
        20125,
        20105,
        44550,
        27193,
        22258,
        20165,
        20101,
        20100,
        33572,
        22661,
        20108,
        24235,
        20011,
        28882,
        20201,
        59599,
        30558,
        20019,
        68731,
        20014,
        20109,
        24853,
        20103,
        20238,
        24878,
        27849,
        20105,
        20100,
        20299,
        20006,
        20549,
        20006,
        20102,
        28808,
        20101,
        25898,
        21195,
        20007,
    ]

    def setup_method(self):
        self.path: str = tempfile.mkdtemp()
        service: TokenizerService = get_tokenizer_service(self.path, BlackHoleCacheBackendConfig())
        self.window_service = WindowServiceFactory.get_window_service("together/glm", service)

    def teardown_method(self, method):
        shutil.rmtree(self.path)

    def test_max_request_length(self):
        assert self.window_service.max_request_length == 2049

    def test_encode(self):
        assert self.window_service.encode(TEST_PROMPT).token_values == TestICEWindowService.TEST_TOKEN_IDS

    def test_decode(self):
        assert self.window_service.decode(self.window_service.encode(TEST_PROMPT).tokens) == TEST_PROMPT

    def test_tokenize(self):
        assert self.window_service.tokenize(TEST_PROMPT) == [
            " The",
            " Center",
            " for",
            " Research",
            " on",
            " Foundation",
            " Models",
            " (",
            "CR",
            "FM",
            ")",
            " is",
            " an",
            " in",
            "terdisciplinary",
            " initiative",
            " born",
            " out",
            " of",
            " the",
            " Stanford",
            " Institute",
            " for",
            " Human",
            "-",
            "Center",
            "ed",
            " Artificial",
            " Intelligence",
            " (",
            "HAI",
            ")",
            " that",
            " aims",
            " to",
            " make",
            " fundamental",
            " advances",
            " in",
            " the",
            " study",
            ",",
            " development",
            ",",
            " and",
            " deployment",
            " of",
            " foundation",
            " models",
            ".",
        ]

    def test_tokenize_and_count(self):
        # There are 52 tokens in `TEST_PROMPT`.
        assert self.window_service.get_num_tokens(TEST_PROMPT) == 50

    def test_fits_within_context_window(self):
        # Should fit in the context window since we subtracted the number of tokens of the test prompt
        # from the max context window
        assert self.window_service.fits_within_context_window(TEST_PROMPT, self.window_service.max_request_length - 50)
        # Should not fit in the context window because we're expecting one more extra token in the completion
        assert not self.window_service.fits_within_context_window(
            TEST_PROMPT, self.window_service.max_request_length - 50 + 1
        )

    def test_truncate_from_right(self):
        # Create a prompt that exceed max context length: 50 * 42 = 2,100 tokens
        long_prompt: str = TEST_PROMPT * 42
        assert not self.window_service.fits_within_context_window(long_prompt)

        # Truncate and ensure it fits within the context window
        truncated_long_prompt: str = self.window_service.truncate_from_right(long_prompt)
        assert self.window_service.get_num_tokens(truncated_long_prompt) == self.window_service.max_request_length
        assert self.window_service.fits_within_context_window(truncated_long_prompt)

    def test_truncate_from_right_with_japanese(self):
        prompt: str = (
            "Passage: "
            "Universe 3 attempt to attack him while meditating, but Jiren's energy alone is enough to push them "
            'back. \n\n\n112\n36\n"A Saiyan\'s Vow! Vegeta\'s Resolution!!"\n"Saiya-jin no chikai! Bejīta no '
            'kakugo!!" (サイヤ人の誓い！ ベジータの覚悟！！) \nOctober 22, 2017\nTBA\n\n\nDespite his exhaustion and low energy, '
            "Goku is determined to keep fighting. However, he is suddenly targeted by Koitsukai (コイツカイ, Koitukai), "
            "Panchia (パンチア, Panchia), and Bollarator (ボラレータ, Borarēta) of Universe 3. Gohan and Piccolo want to help "
            "him, but they are distracted by their own battle with Saonel and Pilina. Caulifla and Kale are "
            "discussing the battle between Goku and Jiren, when they are confronted by Monna (モンナ, Monna), "
            "a female warrior from Universe 4. Caulifla faces off against Monna, but Cabba appears and takes her "
            "place. He instructs Caulifla and Kale to retreat for now. Cabba turns Super Saiyan and battles Monna, "
            "but he is initially overpowered. Monna almost knocks Cabba off the stage, but Vegeta intervenes and "
            "catches him before he can fall. Vegeta and Cabba both promise that, if either of them wins the "
            "tournament, the winner will use the Super Dragon Balls to restore the other's erased universe and its "
            "inhabitants. Newly inspired, Cabba fights Monna again, but he is losing until he taps into his anger and "
            "unlocks Super Saiyan 2. With his new power, Cabba easily defeats and eliminates Monna. However, "
            "Frieza appears, fights Cabba, and easily eliminates him as well. Frieza reveals that he plans to win the "
            "tournament in order to use the Super Dragon Balls to control the gods. Meanwhile, Vegeta decides to "
            "challenge Jiren, but he is intercepted by Toppo. The two battle each other. Vegeta ends up meeting up "
            "with Goku, who is still having difficulty with his three opponents. However, Caulifla intervenes and "
            "knocks Koitsukai, Borareta, and Pancea away before powering up to Super Saiyan 2 and challenging Goku "
            'herself. \n\n\n113\n37\n"With Great Joy! The Repeat Battle-Crazy Saiyan Fight!!"\n"Kiki to Shite! '
            'Sentō-kyō Saiya-jin Batoru Futatabi!!" (嬉々として! 戦闘狂サイヤ人バトル再び!!) \nOctober 29, 2017\nTBA\n\n\nCaulifla '
            "demands that Goku fight her and teach her how to unlock Super Saiyan 3 so that she can become strong "
            "enough to win the tournament. She begins fighting Goku, who does not initially power up beyond his base "
            "form. However, Goku's superior martial arts skills allow him to keep up with her even in her Super "
            "Saiyan 2 form. Eventually, she is able to adapt to his moves, which forces Goku to power up to Super "
            "Saiyan 2 himself. The two of them fight evenly for some time. Goku later calls in Kale to join the "
            "battle. His superior power and skill enables him to fight them both at once, but they are able to match "
            "him by working together. They land a massive combined attack on him, but Goku powers up to Super Saiyan "
            "3 and easily deflects their attacks. However, Goku lacks the stamina to maintain that form and reverts "
            "to Super Saiyan 2. Kale's desire to become stronger inadvertently causes her to transform into her "
            'Berserker Super Saiyan form again. \n\n\n114\n38\n"Intimidating Passion! The Birth of a New Super '
            'Warrior!!"\n"Kiki semaru! Aratana chō sen-shi no bakutan!!" (鬼気せまる! 新たな超戦士の爆誕!!) \nNovember 5, '
            "2017\nTBA\n\n\nVegeta continues his battle with Toppo, while Caulifla is able to help Kale regain "
            "control of her Berserker form. Caulifla and Kale battle Goku together and are able to hold him off. "
            "Meanwhile, Frieza is confronted by Katopesla, but they are interrupted by Goku's battle with Caulifla "
            "and Kale. Frieza wants to fight Caulifla and Kale, but Goku insists that he will fight them himself. "
            "Frieza backs down and watches the fight. The three Saiyans resume their battle. After unsuccessfully "
            "attempting to use Instant Transmission as a strategy, Goku powers up to his Super Saiyan God form and "
            "easily overpowers the two of them. As Goku prepares to eliminate Caulifla and Kale with a powerful "
            "Kamehameha, it is revealed that the girls were given a pair of Potara earrings by Fuwa, Universe 6's "
            "Supreme Kai. They use the earrings and fuse to become a single being with immense power, who takes the "
            'name Kefla (ケフラ, Kefura). \n\n\n115\n39\n"Goku VS Kefla! Super Saiyan Blue Defeated?!"\n"Gokū vāsasu '
            'Kefura! Sūpāsaiya-jin Burū Yabureru!?" (悟空VSケフラ! 超サイヤ人ブルー敗れる!?) \nNovember 12, 2017\nTBA\n\n\nAs Vegeta '
            "continues to fight Toppo, Gohan and Piccolo are engaged with Saonel and Pirina. No. 18 is attacked by "
            "Katopesla, but she is saved by No. 17. Meanwhile, Goku continues his fight with Kefla. The Zenōs approve "
            "Kefla's use of the Potara earrings. The other universes consider giving the Potara to their warriors as "
            "well. Pell (ペル, Peru), the Supreme Kai of Universe 2, gives his Potara to Rabanra (ラバンラ, Rabanra) and "
            "Zarbuto (ザーブト, Zābuto), but the earrings are destroyed by Kefla charging through them to continue "
            "fighting Goku. The battle becomes intense with Kefla gaining the upper hand. Goku powers up to Super "
            "Saiyan Blue, but Kefla counters by powering up to Super Saiyan. However, Goku uses the Kaio-ken and "
            "regains the advantage. He appears to be winning, but Kefla lands a sneak attack that knocks him out of "
            "Super Saiyan Blue. She prepares to eliminate him, but Goku unexpectedly reawakens Ultra Instinct again "
            "and easily dodges her attacks. \n\n\n116\n40\n\"The Sign of a Comeback! Ultra Instinct's Huge "
            'Explosion!!"\n"Gyakuten no kizashi! Migatte no gokui ga dai bakuhatsu!!" (逆転の兆し！ 身勝手の極意が大爆発！！) '
            "\nNovember 19, 2017\nTBA\n\n\n\nTo counter Goku's increase in power, Kefla powers up to Super Saiyan 2, "
            "and the two of them face off. Goku still easily dodges Kefla's attacks, but his own attacks are not "
            "enough to take her down. When Goku launches his attacks, it interferes with his concentration and "
            "prevents him from using Ultra Instinct to its full potential. Jiren senses the energy from their battle, "
            "which prompts him to awaken from his meditation and rejoin Toppo and Dyspo. Vegeta realizes that Ultra "
            "Instinct is the level of skill that Whis was training him and Goku to attain. Vegeta decides that he "
            "must reach it too. Goku begins running low on stamina. He declares that he will end the fight with his "
            "next attack. Kefla panics and unleashes a multitude of deadly energy beams. Her ultimate attack "
            "devastates the ring, but Goku easily dodges her blasts while charging a Kamehameha. Goku jumps into the "
            "air. Kefla focuses all of her power into a single blast and launches it at him. She takes advantage of "
            "his apparent inability to dodge. However, he back flips and uses the charge up energy to slide over her "
            "attack and launches his Kamehameha at point-blank range. Goku blasts Kefla out of the ring and "
            "eliminates her. Her Potara earrings shatter, and she splits back into Kale and Caulifla. With both of "
            "them eliminated, Saonel and Pirina are the only remaining warriors from Team Universe 6.\n\nNOTE: This "
            "episode is dedicated in memory of Hiromi Tsuru, who passed away on November 16, "
            '2017. \n\n\n117\n41\n"Showdown of Love! Androids VS Universe 2!!"\n"Ai no daikessen! Jinzōningen VS Dai '
            'ni uchū!!" (愛の大決戦！人造人間VS第２宇宙！！) \nNovember 26, 2017\nTBA\n\n\nGoku is left exhausted from his battle '
            "with Kefla. He is confronted by all five remaining warriors from Universe 2. Gohan and Piccolo try to "
            "help him, but they are intercepted by Saonel and Pirina. Elsewhere, Vegeta faces off against Katopesla "
            "and tries to unlock Ultra Instinct against him. However, he is unsuccessful and resorts to his normal "
            "fighting style. Vegeta easily overpowers Katopesla. Rozie and Ribrianne launch a combined attack at "
            "Goku, but No. 17 and No. 18 arrive in time to deflect it. The androids battle Rozie and Ribrianne. No. "
            "17 defeats and eliminates Rozie, while No. 18 knocks Ribrianne out of her powered-up transformation. "
            "This causes her to revert into Brianne. Brianne is able to catch No. 18 in an energy trap, "
            "while her comrades from Universe 2 send their love to her. This allows Brianne to transform into a giant "
            "manifestation version of her Super Ribrianne form. No. 18 almost gives up until her love for Krillin and "
            "Marron gives her the determination to break free of Ribrianne's trap. With No. 17's help, "
            "No. 18 blasts through Ribrianne's giant form and eliminates her. Brianne realizes that she lost because "
            "of No. 18's strong love for Krillin, while Goku faces off against Zirloin (ザーロイン, Zāroin), Zarbuto, "
            'and Rabanra of Universe 2. \n\n\n118\n42\n"Accelerated Tragedy Vanishing Universes..."\n"Kasokusuru '
            'Higeki Kieyuku Uchū..." (加速する悲劇消えゆく宇宙...) \nDecember 3, 2017\nTBA\n\n\nGoku battles against Zirloin, '
            "Zarbuto, and Rabanra from Universe 2, while Gohan and Piccolo are fighting against Saonel and Pirina. "
            "Saonel and Pirina's power suddenly increases, and it is revealed that they had each fused with many "
            "Namekians from their universe before coming to the tournament, which greatly increased their own powers. "
            "No. 17 and No. 18 join Goku to help in the battle against the remaining Universe 2 warriors. It is "
            "revealed that Universe 2's angel, Sour (サワア, Sawaa), has been broadcasting the tournament to the "
            "populace of their universe. Universe 2's inhabitants channel their love to Zirloin, Zarbuto, "
            "and Rabanra, who are able to transform into forms similar to Ribrianne, Rozie, and Kakunsa's "
            "transformed states. Goku, No. 17, and No. 18 battle Zirloin, Zarbuto, and Rabanra, while Gohan and "
            "Piccolo continue fighting Saonel and Pirina. The Universe 2 trio unleash a legendary Universe 2 "
            "technique, the Pretty Black Hole, which traps and threatens to sink Goku, No. 17, and No. 18 through the "
            "fighting stage. Goku powers up to Super Saiyan Blue and breaks through the Pretty Black Hole with a "
            "Kamehameha that eliminates Zirloin, Zarbuto, and Rabanra, while Gohan and Piccolo blast Saonel and "
            "Pirina off the ring with a powerful combined attack that eliminates them as well. With all of their "
            "fighters eliminated, both Universe 2 and Universe 6 are erased. Brianne leads the Universe 2 team in a "
            "final happy moment before their erasure, while Cabba wishes Vegeta good luck. Champa taunts Beerus "
            "before being erased. Beerus remains impassive in the face of his brother's erasure. Vegeta faces off "
            'against Katopesla and warns him that he is in a bad mood. \n\n\n119\n43\n"Unavoidable?! The Fierce '
            'Stealth Attack!!"\n"Kaihi funō!? Suterusu kōgeki no mōi!!" (回避不能!? ステルス攻撃の猛威!!) \nDecember 10, '
            "2017\nTBA\n\n\nVegeta overpowers Katopesla and drives him to the edge of the ring. Katopesla is able to "
            "catch himself, but he is pushed over the edge and eliminated by an unseen force. Vegeta and Gohan are "
            "attacked by the same force, but Vegeta is able to stop himself from going over the edge, while Piccolo "
            "catches Gohan. No. 18 is attacked by the invisible fighter, who is revealed to be one of the missing "
            "Universe 4 fighters, Gamisaras (ガミサラス, Gamisarasu). Gohan creates a cloud of dust that coats Gamisaras, "
            "which allows Piccolo to see him and then easily eliminates Gamisaras. Undaunted, Quitela orders the "
            "remaining Universe 4 fighters to step up their game. Shantza (シャンツァ, Shantsa) generates a dome that "
            "envelops the Universe 7 team and manifests illusions of the defeated fighters from the erased universes. "
            "However, Piccolo spots Shantza and blasts him off the ring, which eliminates him and destroys the "
            "illusions. Universe 4's last fighter, Damon (ダモン, Damon), is also assumed to be invisible since no one "
            "can see him. Piccolo is able to sense Damon's attacks, but he proves unable to hit Damon, who knocks "
            "him out of the ring and eliminates him. No. 17 discovers the truth and exposes Damon as a tiny bug-like "
            "creature rather than an invisible person, which explains why none of the fighters were able to hit him. "
            "To solve this problem, Goku repeatedly punches the ring and creates shock waves that launch Damon into "
            "the air, which cancels out his agility and allows No. 17 to hit him with energy blasts. No. 17 traps "
            "Damon in a miniature force field and kicks him out of the ring to eliminate him. With all of their "
            'fighters eliminated, Universe 4 is promptly erased. \n\n\n120\n44\n"The Perfect Survival Tactic! '
            'Universe 3\'s Menacing Assassin!!"\n"Kanpeki na Seizon Senryaku! Dai san Uchū Kyōi no Shikaku!!" ('
            "完璧なる生存戦略! 第３宇宙脅威の刺客!!) \nDecember 17, 2017\nTBA\n\n\nFollowing the elimination of Universe 4, "
            "the remaining fighters from Universe 3 take the offensive against Universe 7. While Viara is defeated "
            "and eliminated by No. 17 and No. 18's combined efforts, Paparoni (パパロニ, Paparoni) sends Panchia, "
            "Koitsukai, and Bollarator to attack Goku, Gohan, and Vegeta. To help Goku and Vegeta save their energy "
            "to confront Universe 11, Gohan decides to face the three robots alone. He gains the upper hand until "
            "Paparoni has them combine together into a much stronger robot called Koichiarator (コイチアレータ, "
            "Koichiarēta). Koichiarator overpowers Gohan until Goku and Vegeta step in. The two distract the enemy, "
            "while Gohan charges and strikes with an attack powerful enough to defeat it. With his robots defeated, "
            "Paparoni refuses to surrender and declares that he will unleash his trump card on Universe 7. "
            "\n\n\n121\n45\n\"All-Out War! The Ultimate Quadruple Merge vs Universe 7's Full-Scale "
            'Attack!!"\n"Sōryokusen! Kyūkoku no Yontai Gattai VS Dai nana Uchū Sōkōgeki!!" (総力戦！究極の4体合体VS第7宇宙総攻撃！！) '
            "\nDecember 24, 2017\nTBA\n\n\nPaparoni and Koichiarator merge to form Anilaza (アニラーザ, Anirāza), "
            "the most powerful warrior from Universe 3. Anilaza begins to overwhelm the Saiyans, which forces No. 17 "
            "and No. 18 to reinforce them. The five attempt to coordinate their attacks to catch Anilaza off-guard, "
            "but he deflects them all. It is revealed that Anilaza can use echolocation to detect his opponents' "
            "movements. Anilaza begins teleporting his punches, and he nearly knocks Goku off the stage until Frieza "
            "steps in and knocks him back into the arena. Realizing that the Universe 7 warriors will rescue each "
            "other from defeat, Anilaza attempts to eat No. 18, who is rescued by Goku. Anilaza corners No. 17 and "
            "overwhelms him. He knocks him off the fighting stage, but No. 18 sacrifices herself to kick No. 17 back "
            "onto the fighting stage. She is eliminated from the tournament. With no other options, the Universe 7 "
            "warriors power up to their maximum levels and engage in a ki clash with Anilaza. They buy time for No. "
            "17 to pierce through Anilaza's attack and damage his energy reactor. This enables the others to "
            "overwhelm and eliminate Anilaza. With all of their warriors eliminated, Universe 3 is erased. As the "
            "Universe 7 warriors begin to celebrate, the remaining Universe 11 warriors step forward to challenge "
            'them. \n\n\n122\n46\n"For One\'s Own Pride! Vegeta\'s Challenge to Be The Strongest!!"\n"Onore no Hokori '
            'wo Kakete! Bejīta Saikyō he no Chōsen!!" (己の誇りをかけて！ベジータ最強への挑戦！！) \nJanuary 7, 2018\nTBA\n\n\nWith only '
            "two universes remaining, the Great Priest compresses the bleachers so that everyone is brought together. "
            "The final warriors begin to fight each other. Gohan and No. 17 battle Toppo. Frieza fights Dyspo. Goku "
            "and Vegeta battle Jiren. Jiren overwhelms Goku with a flurry of punches. Vegeta analyzes his patterns "
            "and dodges his attacks. He lands a solid blow to Jiren's midsection. Jiren counters with a powerful "
            "blast that nearly rings Vegeta out. Meanwhile, Frieza blocks one of Dyspo's attacks with his tail, "
            "but Dyspo uses it as leverage to injure him. Jiren disparages Vegeta for his self-righteousness, "
            "but Vegeta declares that it is the source of his strength. He powers up a Final Flash and goads Jiren "
            "into taking it head-on. However, the attack fails to damage Jiren, who acknowledges the power of "
            "Vegeta's attack before incapacitating him. \n\n\n123\n47\n\"Body and Soul, Full Power Release! Goku and "
            'Vegeta!!"  \nJanuary 14, 2018\nTBA\n\n\n\nQuestion: When does dragon ball super episode 113 '
            "start?\nAnswer:"
        )
        max_completion_length: int = 300
        desired_length: int = 1749

        truncated_prompt: str = self.window_service.truncate_from_right(prompt, max_completion_length)
        truncated_length: int = self.window_service.get_num_tokens(truncated_prompt)
        assert truncated_length == desired_length, f"Should be {desired_length} long, but was {truncated_length} long."
