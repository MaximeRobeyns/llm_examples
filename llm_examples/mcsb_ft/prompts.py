# Copyright (C) 2023 Maxime Robeyns <dev@maximerobeyns.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Prompts for the MCSB task"""

description_prompt = """Write a description for each of the following wods:

Word: satisfaction
Description: This is a pleasant feeling often associated with a sense of accomplishment.

Word: nightmare
Description: An unpleasant dream that one often wakes up from with cold sweat.

Word: coordination
Description: The act of working together as a team and communicating effectively.

Word: {}
Description:"""

question_preamble = (
    "Return the label of the word which best matches the description.\n\n"
)
