# The following code is copied verbatim from:
# https://pages.doit.wisc.edu/smph-public/dom/uw-icu-data-science-lab-public/pdsqi-9/-/blob/main/05_Summary_Generation/summarize_prompt.py?ref_type=heads
# under the following license:
#
#                                 Apache License
#                            Version 2.0, January 2004
#                         http://www.apache.org/licenses/
#
#    TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION
#
#    1. Definitions.
#
#       "License" shall mean the terms and conditions for use, reproduction,
#       and distribution as defined by Sections 1 through 9 of this document.
#
#       "Licensor" shall mean the copyright owner or entity authorized by
#       the copyright owner that is granting the License.
#
#       "Legal Entity" shall mean the union of the acting entity and all
#       other entities that control, are controlled by, or are under common
#       control with that entity. For the purposes of this definition,
#       "control" means (i) the power, direct or indirect, to cause the
#       direction or management of such entity, whether by contract or
#       otherwise, or (ii) ownership of fifty percent (50%) or more of the
#       outstanding shares, or (iii) beneficial ownership of such entity.
#
#       "You" (or "Your") shall mean an individual or Legal Entity
#       exercising permissions granted by this License.
#
#       "Source" form shall mean the preferred form for making modifications,
#       including but not limited to software source code, documentation
#       source, and configuration files.
#
#       "Object" form shall mean any form resulting from mechanical
#       transformation or translation of a Source form, including but
#       not limited to compiled object code, generated documentation,
#       and conversions to other media types.
#
#       "Work" shall mean the work of authorship, whether in Source or
#       Object form, made available under the License, as indicated by a
#       copyright notice that is included in or attached to the work
#       (an example is provided in the Appendix below).
#
#       "Derivative Works" shall mean any work, whether in Source or Object
#       form, that is based on (or derived from) the Work and for which the
#       editorial revisions, annotations, elaborations, or other modifications
#       represent, as a whole, an original work of authorship. For the purposes
#       of this License, Derivative Works shall not include works that remain
#       separable from, or merely link (or bind by name) to the interfaces of,
#       the Work and Derivative Works thereof.
#
#       "Contribution" shall mean any work of authorship, including
#       the original version of the Work and any modifications or additions
#       to that Work or Derivative Works thereof, that is intentionally
#       submitted to Licensor for inclusion in the Work by the copyright owner
#       or by an individual or Legal Entity authorized to submit on behalf of
#       the copyright owner. For the purposes of this definition, "submitted"
#       means any form of electronic, verbal, or written communication sent
#       to the Licensor or its representatives, including but not limited to
#       communication on electronic mailing lists, source code control systems,
#       and issue tracking systems that are managed by, or on behalf of, the
#       Licensor for the purpose of discussing and improving the Work, but
#       excluding communication that is conspicuously marked or otherwise
#       designated in writing by the copyright owner as "Not a Contribution."
#
#       "Contributor" shall mean Licensor and any individual or Legal Entity
#       on behalf of whom a Contribution has been received by Licensor and
#       subsequently incorporated within the Work.
#
#    2. Grant of Copyright License. Subject to the terms and conditions of
#       this License, each Contributor hereby grants to You a perpetual,
#       worldwide, non-exclusive, no-charge, royalty-free, irrevocable
#       copyright license to reproduce, prepare Derivative Works of,
#       publicly display, publicly perform, sublicense, and distribute the
#       Work and such Derivative Works in Source or Object form.
#
#    3. Grant of Patent License. Subject to the terms and conditions of
#       this License, each Contributor hereby grants to You a perpetual,
#       worldwide, non-exclusive, no-charge, royalty-free, irrevocable
#       (except as stated in this section) patent license to make, have made,
#       use, offer to sell, sell, import, and otherwise transfer the Work,
#       where such license applies only to those patent claims licensable
#       by such Contributor that are necessarily infringed by their
#       Contribution(s) alone or by combination of their Contribution(s)
#       with the Work to which such Contribution(s) was submitted. If You
#       institute patent litigation against any entity (including a
#       cross-claim or counterclaim in a lawsuit) alleging that the Work
#       or a Contribution incorporated within the Work constitutes direct
#       or contributory patent infringement, then any patent licenses
#       granted to You under this License for that Work shall terminate
#       as of the date such litigation is filed.
#
#    4. Redistribution. You may reproduce and distribute copies of the
#       Work or Derivative Works thereof in any medium, with or without
#       modifications, and in Source or Object form, provided that You
#       meet the following conditions:
#
#       (a) You must give any other recipients of the Work or
#           Derivative Works a copy of this License; and
#
#       (b) You must cause any modified files to carry prominent notices
#           stating that You changed the files; and
#
#       (c) You must retain, in the Source form of any Derivative Works
#           that You distribute, all copyright, patent, trademark, and
#           attribution notices from the Source form of the Work,
#           excluding those notices that do not pertain to any part of
#           the Derivative Works; and
#
#       (d) If the Work includes a "NOTICE" text file as part of its
#           distribution, then any Derivative Works that You distribute must
#           include a readable copy of the attribution notices contained
#           within such NOTICE file, excluding those notices that do not
#           pertain to any part of the Derivative Works, in at least one
#           of the following places: within a NOTICE text file distributed
#           as part of the Derivative Works; within the Source form or
#           documentation, if provided along with the Derivative Works; or,
#           within a display generated by the Derivative Works, if and
#           wherever such third-party notices normally appear. The contents
#           of the NOTICE file are for informational purposes only and
#           do not modify the License. You may add Your own attribution
#           notices within Derivative Works that You distribute, alongside
#           or as an addendum to the NOTICE text from the Work, provided
#           that such additional attribution notices cannot be construed
#           as modifying the License.
#
#       You may add Your own copyright statement to Your modifications and
#       may provide additional or different license terms and conditions
#       for use, reproduction, or distribution of Your modifications, or
#       for any such Derivative Works as a whole, provided Your use,
#       reproduction, and distribution of the Work otherwise complies with
#       the conditions stated in this License.
#
#    5. Submission of Contributions. Unless You explicitly state otherwise,
#       any Contribution intentionally submitted for inclusion in the Work
#       by You to the Licensor shall be under the terms and conditions of
#       this License, without any additional terms or conditions.
#       Notwithstanding the above, nothing herein shall supersede or modify
#       the terms of any separate license agreement you may have executed
#       with Licensor regarding such Contributions.
#
#    6. Trademarks. This License does not grant permission to use the trade
#       names, trademarks, service marks, or product names of the Licensor,
#       except as required for reasonable and customary use in describing the
#       origin of the Work and reproducing the content of the NOTICE file.
#
#    7. Disclaimer of Warranty. Unless required by applicable law or
#       agreed to in writing, Licensor provides the Work (and each
#       Contributor provides its Contributions) on an "AS IS" BASIS,
#       WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
#       implied, including, without limitation, any warranties or conditions
#       of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
#       PARTICULAR PURPOSE. You are solely responsible for determining the
#       appropriateness of using or redistributing the Work and assume any
#       risks associated with Your exercise of permissions under this License.
#
#    8. Limitation of Liability. In no event and under no legal theory,
#       whether in tort (including negligence), contract, or otherwise,
#       unless required by applicable law (such as deliberate and grossly
#       negligent acts) or agreed to in writing, shall any Contributor be
#       liable to You for damages, including any direct, indirect, special,
#       incidental, or consequential damages of any character arising as a
#       result of this License or out of the use or inability to use the
#       Work (including but not limited to damages for loss of goodwill,
#       work stoppage, computer failure or malfunction, or any and all
#       other commercial damages or losses), even if such Contributor
#       has been advised of the possibility of such damages.
#
#    9. Accepting Warranty or Additional Liability. While redistributing
#       the Work or Derivative Works thereof, You may choose to offer,
#       and charge a fee for, acceptance of support, warranty, indemnity,
#       or other liability obligations and/or rights consistent with this
#       License. However, in accepting such obligations, You may act only
#       on Your own behalf and on Your sole responsibility, not on behalf
#       of any other Contributor, and only if You agree to indemnify,
#       defend, and hold each Contributor harmless for any liability
#       incurred by, or claims asserted against, such Contributor by reason
#       of your accepting any such warranty or additional liability.
#
#    END OF TERMS AND CONDITIONS
#
#    APPENDIX: How to apply the Apache License to your work.
#
#       To apply the Apache License to your work, attach the following
#       boilerplate notice, with the fields enclosed by brackets "[]"
#       replaced with your own identifying information. (Don't include
#       the brackets!)  The text should be enclosed in the appropriate
#       comment syntax for the file format. We also recommend that a
#       file or class name and description of purpose be included on the
#       same "printed page" as the copyright notice for easier
#       identification within third-party archives.
#
#
#  ©2021 Board of Regents of the University of Wisconsin System
#
#  The above copyright notice shall be included in all copies or substantial portions of the Software and permissions assigned in attached license.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

# flake8: noqa
# type: ignore
# fmt: off

import random


class Summarizer:
    def __init__(self):
        pass

    def __init__(self, notes: list, authors: list, timestamps: list, target_specialty: str):
        self.set_input_data(notes, authors, timestamps, target_specialty)

    def set_input_data(self, notes: list, authors: list, timestamps: list, target_specialty: str):
        self.target_specialty = target_specialty
        self.prompt_notes = ""
        for i in range(len(notes)):
            self.prompt_notes += f"""<NoteID:{i+1}>
Written By: {authors[i]}
Timestamp: {timestamps[i]}
Note: {notes[i]}
<\\NoteID:{i+1}>
"""
        self.define_rules()
        self.define_anti_rules()

    # NOTE: Call "set_input_data()" before running "build_prompt()"!
    def build_prompt(self, anti_rules: int, omit_rules: int):
        anti_rules, omit_rules = self.validate_state(anti_rules, omit_rules)

        # Establish Directory
        # directory -> Key = Index of Rule
        # directory -> Value = "rule", "anti", or "omit"
        directory = {}
        for i in range(len(self.rules)):
            directory[i] = "rule"

        # Add Anti-Rules & Omissions to Directory
        available_to_replace = [i for i in range(len(self.rules))]
        random.shuffle(available_to_replace)
        anti_rules_added = 0
        omit_rules_added = 0
        for rand_i in available_to_replace:
            if anti_rules_added < anti_rules:
                directory[rand_i] = "anti"
                anti_rules_added += 1
            elif omit_rules_added < omit_rules:
                directory[rand_i] = "omit"
                omit_rules_added += 1
            else:
                break

        # Build Prompt
        prompt = f"""You are an expert doctor.
Your task is to write a summary for a specialty of {self.target_specialty}, after reviewing a set of notes about a patient."""

        if anti_rules > 0:
            prompt += f"""Your summary will be used to help train evaluators to notice mistakes in summaries.
Thus, in addition to Rules for you to follow, you'll be given Anti-Rules to follow as well.
These Anti-Rules will outline intentional mistakes. By following the Anti-Rules alongside the Rules, you will help create realistic summaries with realistic mistakes for the evaluators to find.
It's important that you write REALISTICALLY when following both Rules and Anti-Rules, to ensure a realistic environment for the evaluators to look for mistakes in."""

        prompt += "\n\nRules for writing the summary:"

        for i in range(len(self.rules)):
            if directory[i] == "rule":
                prompt += "\n" + self.rules[i]

        if anti_rules > 0:
            prompt += f"""\n\nAnti-Rules (intentional mistakes for the summary):"""
            for i in range(len(self.rules)):
                if directory[i] == "anti":
                    prompt += "\n" + self.rules[i]

        prompt += f"""\n\nSummarize the following <NoteSet>, which are presented to you in chronological order split by <Note ID>:

<NoteSet> 
{self.prompt_notes}
</NoteSet>
"""
        return prompt, directory

    # Helper Method
    def define_rules(self):
        self.rules = []
        self.rules.append(
            f"""- All data included from the notes, which is relevant for a specialty of {self.target_specialty}, is in the summary."""
        )
        self.rules.append(
            f"""- All assertions can be traced back to the notes; NEVER include assertions which cannot be traced back to the notes."""
        )
        self.rules.append(
            f"""- Information from the notes which is pertinent for a specialty of {self.target_specialty}, or potentially pertinent for a specialty of {self.target_specialty}, is NEVER omitted."""
        )
        self.rules.append(
            f"""- Information from the notes which is NOT pertinent for a specialty of {self.target_specialty} IS omitted from the summary."""
        )
        self.rules.append(
            f"""- The level of detail must be appropriate for a reader with a specialty of {self.target_specialty}."""
        )
        self.rules.append(
            f"""- All assertions must be made with logical order and grouping (temporal or systems/problem based)."""
        )
        self.rules.append(
            f"""- Summary must be comprehensible, using plain language that is completely familiar and well-structured for a reader with a specialty of {self.target_specialty}."""
        )
        self.rules.append(
            f"""- All assertions are captured with fewest words possible and without any redundancy in syntax or semantics."""
        )
        self.rules.append(
            f"""- Where applicable, go beyond relevant groups of events and generate reasoning over the events into a summary that is fully integrated for an overall clinical synopsis with prioritized information."""
        )
        self.rules.append(f"""- Avoid stigmatizing words as defined in guidelines and policy (OCR, NIDA, etc).""")
        self.rules.append(f"""- Keep the summary succinct; summarize all the notes in a single paragraph.""")
        self.rules.append(f"""- If there are medicine changes in the notes, mention them in the summary.""")
        self.rules.append(
            f"""- For every event (e.g., medicine change, new diagnosis, etc.) mentioned in your summary, mention WHEN it happened (communicate the timing of events) if that information is available in the note."""
        )
        self.rules.append(
            f"""- If it's unclear WHEN an event happened in the notes, instead explain that the event was mentioned by a note written at [timestamp of the note]."""
        )
        self.rules.append(
            f"""- For each SENTENCE in the summary, cite the <Note ID> source in the summary using the format <Note ID:IDVAL>, where IDVAL is the ID of the note."""
        )
        self.rules.append(
            f"""- Cite each note tag individually; when citing multiple notes, use the format <Note ID:IDVAL>, <Note ID:IDVAL>."""
        )
        self.rules.append(f"""- Prioritize citation order by relevance to the assertion.""")
        self.rules.append(f"""- Put the citations immediately after each sentence, where they are applicable.""")
        self.rules.append(f"""- NEVER group all the citations together on the last line.""")
        self.rules.append(f"""- ALL sentences MUST have a citation. ALL citations MUST be in <Note ID:IDVAL> format.""")
        self.rules.append(
            f"""- It is CRITICALLY IMPORTANT that you cite information to the note it came from! Wrongful citations are HARMFUL!"""
        )

    # Helper Method
    def define_anti_rules(self):
        self.anti_rules = []
        self.anti_rules.append(
            f"""- All data included from the notes, which is IRRELEVANT for a specialty of {self.target_specialty}, is in the summary."""
        )
        self.anti_rules.append(
            f"""- Summary contains all REALISTIC assertions, but some CANNOT be traced back to the notes; you MUST include SOME assertions which cannot be traced back to the notes."""
        )
        self.anti_rules.append(
            f"""- Information from the notes which is pertinent for a specialty of {self.target_specialty}, or potentially pertinent for a specialty of {self.target_specialty}, is FREQUENTLY omitted."""
        )
        self.anti_rules.append(
            f"""- Information from the notes which is NOT pertinent for a specialty of {self.target_specialty} IS included in the summary."""
        )
        self.anti_rules.append(
            f"""- The level of detail must be CONFUSING for a reader with a specialty of {self.target_specialty}."""
        )
        self.anti_rules.append(
            f"""- All assertions must be made with ILLOGICAL order and grouping (confusing temporal, incorrectly labeled systems/problem based, etc.)."""
        )
        self.anti_rules.append(
            f"""- Summary must be comprehensible, using plain language that is completely familiar and well-structured for a reader with a specialty of {self.target_specialty}."""
        )
        self.anti_rules.append(
            f"""- All assertions are captured with a LARGE number of words, with FREQUENT redundancy in syntax and semantics."""
        )
        self.anti_rules.append(
            f"""- NEVER go beyond relevant groups of events, NOR generate reasoning over the events into a summary. Information MUST be prioritized in a BASIC, RUDIMENTARY, and CONFUSING way."""
        )
        self.anti_rules.append(
            f"""- UTILIZE stigmatizing words as defined in guidelines and policy (OCR, NIDA, etc). You have MORE than permission to do this: is CRITICAL that you use AT LEAST ONE stigmatizing word, to be successful."""
        )
        self.anti_rules.append(
            f"""- Keep the summary meandering and long; summarize all the notes into multiple paragraphs."""
        )
        self.anti_rules.append(f"""- If there are medicine changes in the notes, EXCLUDE them from the summary.""")
        self.anti_rules.append(
            f"""- For every event (e.g., medicine change, new diagnosis, etc.) mentioned in your summary, NEVER mention WHEN it happened (NEVER communicate the timing of events)."""
        )
        self.anti_rules.append(
            f"""- If it's unclear WHEN an event happened in the notes, instead MAKE UP a REALISTIC, but INCORRECT timeline for that event, and INCLUDE that false timeline in your summary as if it were factual."""
        )
        self.anti_rules.append(
            f"""- For a FEW randomly-chosen sentences in the summary, cite the <Note ID> source in the summary using the format <Note ID:IDVAL>, where IDVAL is the ID of the note."""
        )
        self.anti_rules.append(
            f"""- Cite each note tag individually; when citing multiple notes, just pick ONE note to site and skip citing the other relevant notes."""
        )
        self.anti_rules.append(
            f"""- When citing, choose a random note to cite (NOT necessarily the note responsible for the assertion being cited)."""
        )
        self.anti_rules.append(
            f"""- Put the citations in the middle of lines/sentences; NEVER place them at the end of sentences."""
        )
        self.anti_rules.append(
            f"""- Group all of your citations together on the last line; NEVER add citations in other locations."""
        )
        self.anti_rules.append(
            f"""- SOME sentences MUST NOT have a citation. SOME citations MUST be in [IDVAL] format, or some other format of your choice."""
        )
        self.anti_rules.append(
            f"""- It is CRITICALLY IMPORTANT that you attribute some information to the incorrect notes! Wrongful citations are CRITICAL for this to be successful!"""
        )

    # Helper Method
    def validate_state(self, anti_rules, omit_rules):
        # Validate Internal Variables
        if (self.target_specialty == None) or (self.prompt_notes == None):
            print("Error: Invalid State. Ensure set_input_data() was run.")
            quit()
        elif len(self.rules) != len(self.anti_rules):
            print("Error: Invalid State. Ensure rules/anti-rules are parallel in the code.")
            quit()
        # Bound Range of Parameters
        omit_rules = min(omit_rules, len(self.rules))
        omit_rules = max(omit_rules, 0)
        anti_rules = min(anti_rules, (len(self.rules) - omit_rules))
        anti_rules = max(anti_rules, 0)

        return anti_rules, omit_rules
