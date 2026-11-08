# Answer benchmark report

- Suite: `eval/answer_cases.json` version 1
- Generation: google:gemini-3.8-flash; answer prompt answer-v3; summary prompt 4
- Judge: openai:gpt-6.1-sol
- Corpus: story:01aa9b90dbbf summaries:6376951f185b; code cdde5e6-dirty
- Costs are estimates from configured list prices, not provider invoices. Embedding calls are not metered.

## Summary by mode

| Metric | off | heuristic | llm_router | agentic (before fallback fix) | agentic (after fallback fix) |
|---|---|---|---|---|---|
| Cases (judged) | 22 (22) | 22 (22) | 22 (22) | 22 (22) | 22 (22) |
| Pass rate | 45% | 45% | 36% | 64% | 77% |
| Mean completeness | 66% | 61% | 54% | 78% | 92% |
| No contradictions/forbidden claims | 95% | 91% | 91% | 100% | 100% |
| No unsupported statements | 55% | 45% | 64% | 94% | 91% |
| Abstention appropriate | 70% | 70% | 80% | 100% | 100% |
| Required evidence seen | 89% | 89% | 89% | 95% | 89% |
| Required evidence in answer context | 47% | 37% | 42% | 95% | 89% |
| Citations valid | 85% | 74% | 69% | 95% | 100% |
| Temporal leakage | 100% | 100% | 100% | 100% | 0% |
| Errors | 0 | 0 | 0 | 0 | 0 |
| Agent stopped without answering | 0 | 0 | 0 | 4 | 0 |
| Mean latency | 18.8s | 100.2s | 15.1s | 22.5s | 24.5s |
| Model requests | 22 | 22 | 41 | 103 | 103 |
| Tokens in/out | 460,542/49,690 | 440,892/48,533 | 467,591/38,636 | 1,358,348/36,294 | 1,411,054/38,967 |
| Est. system cost | $0.5317 | $0.5127 | unavailable | $1.1549 | $1.2044 |
| Est. judge cost | $0.7591 | $0.7070 | $0.6611 | $1.0182 | $0.9383 |

## Evidence vs answer

Whether every required passage reached the answer context, against whether the answer passed.

| Mode | evidence ✓ pass | evidence ✓ fail | evidence ✗ pass | evidence ✗ fail |
|---|---|---|---|---|
| off | 8 | 1 | 0 | 10 |
| heuristic | 7 | 0 | 1 | 11 |
| llm_router | 6 | 2 | 1 | 10 |
| agentic (before fallback fix) | 12 | 6 | 0 | 1 |
| agentic (after fallback fix) | 14 | 3 | 0 | 2 |

## Pass rate by category

| Category | off | heuristic | llm_router | agentic (before fallback fix) | agentic (after fallback fix) |
|---|---|---|---|---|---|
| alias | 60% | 80% | 40% | 60% | 80% |
| causal | 38% | 25% | 12% | 50% | 62% |
| chronology | 0% | 50% | 50% | 100% | 100% |
| count | 50% | 50% | 50% | 100% | 100% |
| fact | 67% | 50% | 50% | 100% | 100% |
| interleave | 0% | 0% | 0% | 0% | 0% |
| quote | 67% | 67% | 67% | 67% | 100% |
| relationship_change | 33% | 0% | 0% | 67% | 67% |
| unanswerable | 100% | 100% | 50% | 0% | 100% |

## Cases

| Case | off | heuristic | llm_router | agentic (before fallback fix) | agentic (after fallback fix) |
|---|---|---|---|---|---|
| a001_kaho_sayaka_first_meeting | PASS / ev ✓ | PASS / ev ✓ | PASS / ev ✓ | PASS / ev ✓ | PASS / ev ✓ |
| a002_reika_san_identity | partial 75% / ev ✓ | PASS / ev ✓ | partial 25% / ev ✗ | partial 75% / ev ✓ | partial 50% / ev ✓ |
| a003_kaho_transfer_papers | partial 0% / ev ✗ | partial 0% / ev ✗ | partial 0% / ev ✗ | NO ANSWER | partial 67% / ev ✗ |
| a004_burupa_union | partial 0% / ev ✗ | partial 0% / ev ✗ | partial 0% / ev ✗ | PASS / ev ✓ | PASS / ev ✓ |
| a005_sachi_bgp_role | PASS / ev ✓ | PASS / ev ✓ | PASS / ev ✓ | PASS / ev ✓ | PASS / ev ✓ |
| a006_joint_presidency | PASS / ev ✓ | partial 50% / ev ✗ | partial 33% / ev ✗ | PASS / ev ✓ | PASS / ev ✓ |
| a007_link_to_the_future | PASS / ev ✓ | PASS / ev ✓ | partial 17% / ev ✗ | PASS / ev ✓ | PASS / ev ✓ |
| a008_kosuzu_solo_live | WRONG / ev ✗ | partial 0% / ev ✗ | WRONG / ev ✗ | PASS / ev ✓ | PASS / ev ✓ |
| a009_tsuzuri_job_experiences | partial 67% / ev ✗ | partial 67% / ev ✗ | partial 50% / ev ✓ | PASS / ev ✓ | PASS / ev ✓ |
| a010_kozue_invites_kaho | partial 25% / ev ✗ | partial 0% / ev ✗ | partial 0% / ev ✗ | PASS / ev ✓ | PASS / ev ✓ |
| a011_third_year_career_plans | PASS / ev ✓ | WRONG / ev ✗ | PASS / ev ✓ | PASS / ev ✓ | PASS / ev ✓ |
| a012_tsuzuri_advice_quote | PASS / ev ✓ | PASS / ev ✓ | PASS / ev ✓ | PASS / ev ✓ | PASS / ev ✓ |
| a013_unnamed_fan | PASS / ev ✓ | PASS / ev ✓ | PASS / ev ✓ | NO ANSWER | PASS / ev ✓ |
| a014_reika_surname | PASS | PASS | partial 0% | NO ANSWER | PASS |
| a015_reika_line_count | PASS | PASS | PASS | PASS | PASS |
| a016_kosuzu_turn_count | partial 0% | WRONG | WRONG | PASS | PASS |
| a017_omicho_market_history | partial 33% / ev ✗ | partial 33% / ev ✗ | partial 50% / ev ✗ | partial 83% / ev ✓ | partial 83% / ev ✓ |
| a018_ginko_new_tradition | partial 83% / ev ✗ | partial 83% / ev ✗ | partial 83% / ev ✓ | partial 83% / ev ✓ | partial 67% / ev ✓ |
| a019_kaho_sayaka_arc | partial 75% / ev ✗ | partial 25% / ev ✗ | partial 50% / ev ✗ | partial 75% / ev ✓ | partial 50% / ev ✗ |
| a020_episode1_order | partial 17% / ev ✗ | PASS / ev ✗ | PASS / ev ✗ | PASS / ev ✓ | PASS / ev ✓ |
| a021_sayaka_presidency_false_premise | partial 75% / ev ✗ | partial 75% / ev ✗ | partial 75% / ev ✗ | PASS / ev ✓ | PASS / ev ✓ |
| a022_tsuzuri_not_a_school_idol | PASS / ev ✓ | PASS / ev ✓ | PASS / ev ✓ | NO ANSWER | PASS / ev ✓ |

## Failure notes

### a002_reika_san_identity (off)
- just_standing: partial — The answer describes Tsuzuri previously just standing and later working during her placement, but omits Reika agreeing to her request to do proper work in 104 Episode 8.

### a003_kaho_transfer_papers (off)
- papers: missing — The answer does not describe her classmates seeing the transfer papers or worrying that she will leave.
- stays: missing — The answer omits her gathering her classmates, disposing of the papers, and announcing that she will stay.
- change: missing — The answer reports Kozue's suggestion but not Kaho's admission of blaming Hasunosora or her own vow to change the school.
- unsupported: Kaho ultimately decides to join the club alongside Sayaka because she is inspired by Kozue and Tsuzuri; the supplied passages establish her later membership but not this motivation or a joint decision with Sayaka.
- abstention mismatch (abstained=True)

### a004_burupa_union (off)
- name: missing — The answer does not identify the union or its full name.
- why: missing — The answer omits the sequence from December Starring Bloom to nationwide parties and the need for additional organizers.
- members: missing — The answer does not describe the people inspired by Starring Bloom who joined the union.
- abstention mismatch (abstained=True)

### a008_kosuzu_solo_live (off)
- song: contradicted — The answer denies that Kosuzu gives an impromptu solo live and substitutes the earlier rehearsal, never identifying Sparkly Spot or the revenge live.
- stumbles: contradicted — The answer says Kosuzu performs without falling or making mistakes rather than describing her persistence through falls and errors at the solo live.
- sayaka: contradicted — The answer substitutes Sayaka's earlier negative reaction for her affirmation that Kosuzu's feelings reached her, saved her past, and inspired her promise to remain Kosuzu's goal.
- unsupported: There is no impromptu solo live by Kosuzu in 104 Episode 2.
- unsupported: Kosuzu flees in tears after the confrontation; the retrieved passage supports her distress and departure but does not establish tears.

### a009_tsuzuri_job_experiences (off)
- city_hall: partial — The answer identifies city hall clerical work but omits data entry and explicitly declines to identify the dorm mother as arranger.
- market: partial — The answer identifies the market work with Reika-san but does not state that Tsuzuri arranged it herself by asking Reika-san.
- unsupported: The answer describes purchases as including simmered fish, whereas the passage only has Tsuzuri encourage the customer to make a tasty simmered dish.
- abstention mismatch (abstained=True)

### a010_kozue_invites_kaho (off)
- manager: missing — The answer describes an invitation to watch a performance but never identifies Kaho's invited role as club manager.
- idol: partial — The answer conveys Kozue's wish for Kaho to join the club but omits her admission that she wanted to invite Kaho as a school idol too.
- unsupported: The English rendering of Kozue's invitation as showing Kozue a world she does not know reverses the Japanese, which says Kozue will show Kaho a world Kaho does not know.
- abstention mismatch (abstained=True)

### a016_kosuzu_turn_count (off)
- count: missing — The answer declines to give the episode total and reports only a partial-text count rather than the required exact total of 148.
- unsupported: The episode summary references subsequent parts/scenes covering the Tsuruga festival performance.
- abstention mismatch (abstained=True)

### a017_omicho_market_history (off)
- sachi: missing — The answer omits Sachi bringing the members to help in shops, her explanation about local support, and her own senior having brought her.
- live: partial — It mentions an outdoor performance near the market but omits the small park stage and the market staff audience.
- main: partial — It identifies Tsuzuri bringing Sayaka in 103 Episode 5 but omits their helping there and the entire 104 Episode 8 return.
- unsupported: The market performance featured the song Sparkly Spot and began easing Tsuzuri's longstanding dance insecurities.
- unsupported: Sayaka's 103 Episode 5 visit occurred while she faced an expressive plateau and pressure from imitating her sister and senior.
- unsupported: The market visit helped Sayaka break through her artistic block and uncover her authentic sparkle.
- unsupported: Ceras and Izumi are Mizukawa Girls' School transfer students.
- unsupported: Kosuzu mentors her part-time coworker Hiromi to overcome past trauma and severe stage fright.
- unsupported: Hiromi gains confidence by performing at the Small Triangle School Idol Carnival, then resumes working at Omicho Market and assists the Bloom Garden Party Coalition.

### a018_ginko_new_tradition (off)
- festival: partial — The answer describes merging Bloom Days into a lasting Bloom Garden Party tradition but omits the explicit goal of a new Kanazawa festival lasting years or decades and holding it this school year.

### a019_kaho_sayaka_arc (off)
- end: partial — The answer states that they become co-presidents but omits Sayaka's offer to support Kaho as president before Kaho proposes sharing the role.
- unsupported: At their first meeting, Kaho declares that she and Sayaka will quickly become friends; the passage instead says Sayaka will quickly make many friends.
- unsupported: Kaho and Sayaka uncover their seniors' past Love Live! trauma and organize a four-person Nadeshiko Festival performance that mends Kozue and Tsuzuri's rift.
- unsupported: During 104, Sayaka mentors Kosuzu in DOLLCHESTRA.
- unsupported: The national Love Live! victory is specifically a championship playoff victory.
- unsupported: Kozue, Tsuzuri, and Megumi graduate at the Renge Festival.

### a020_episode1_order (off)
- first: partial — The answer identifies the bus meeting at the start of term but does not establish it as first among the three events.
- second: missing — The answer does not state that the classmates see the transfer papers second.
- third: missing — The answer does not state that Kaho announces her decision to stay and change the school last.
- abstention mismatch (abstained=True)

### a021_sayaka_presidency_false_premise (off)
- premise: partial — The answer rejects the skating premise and describes her request for time, but omits her belief that someone with a big dream should lead.
- unsupported: Sayaka served as co-president throughout the entire 105th term.
- unsupported: Sayaka led initiatives such as the Bloom Garden Party.
- unsupported: Sayaka graduated and passed the club presidency to Ginko Momose for the 106th term.

### a003_kaho_transfer_papers (heuristic)
- papers: missing — The answer does not mention classmates discovering the transfer papers or worrying that Kaho will leave.
- stays: missing — The answer declines to describe her handling of the papers and omits her announcement that she will not quit school.
- change: missing — The answer omits both her admission of blaming Hasunosora and her vow to change the school herself.
- unsupported: Kaho ultimately decides to join the School Idol Club alongside Sayaka Murano.
- abstention mismatch (abstained=True)

### a004_burupa_union (heuristic)
- name: missing — The answer does not identify the union or its full name.
- why: missing — The answer omits the sequence from December Starring Bloom to nationwide parties and the need for organizers beyond the club.
- members: missing — The answer does not describe the members as people inspired by Starring Bloom to participate themselves.
- abstention mismatch (abstained=True)

### a006_joint_presidency (heuristic)
- sayaka_proposes: missing — The answer omits Sayaka's belief that a big dream should guide the club and her proposal that Kaho become president.
- joint: partial — The answer identifies Kaho and Sayaka as co-presidents but omits Kaho's counterproposal and Sayaka's agreement.

### a008_kosuzu_solo_live (heuristic)
- song: missing — The answer does not identify Sparkly Spot or describe Kosuzu's revenge live before a crowd.
- stumbles: missing — The answer describes an earlier successful practice but not her falls, mistakes, and perseverance at the solo live.
- sayaka: missing — The answer omits Sayaka's acceptance of Kosuzu's feelings, her sense that her past was saved, and her promise to remain Kosuzu's goal.
- unsupported: The Year 104 overview identifies the new DOLLCHESTRA song as "Ladybug".
- abstention mismatch (abstained=True)

### a009_tsuzuri_job_experiences (heuristic)
- city_hall: partial — The answer identifies city-hall clerical work but omits data entry and explicitly leaves the dorm mother's arranging role unanswered.
- market: partial — The answer identifies market work but fails to identify it as Reika's shop or explain that Tsuzuri arranged it by asking her.
- unsupported: The market is specifically Omicho Market.
- unsupported: Tsuzuri gives change to Reika-san as a customer.
- abstention mismatch (abstained=True)

### a010_kozue_invites_kaho (heuristic)
- manager: missing — The answer abstains without identifying manager as the role Kozue stated.
- idol: missing — The answer never states that Kozue actually wanted to invite Kaho as a school idol too.
- abstention mismatch (abstained=True)

### a011_third_year_career_plans (heuristic)
- sayaka: contradicted — The answer denies Sayaka has a concrete plan instead of stating her university and figure-skating plans at the specified point.
- kaho: missing — The answer claims insufficient information rather than giving Kaho's university plan and uncertainty about her specific goals.
- rurino: missing — The answer omits Rurino's probable decision against university and her uncertainty about following Megumi.
- unsupported: Sayaka has no concrete post-graduation plan in Episode 7.
- unsupported: Later in Year 105, Sayaka plans broad university studies in sports science and coaching.
- unsupported: By the end of Year 105, Kaho passes university entrance examinations to study event production.
- unsupported: Kaho's university plan is not discussed or mentioned in Episode 7.
- abstention mismatch (abstained=True)

### a016_kosuzu_turn_count (heuristic)
- count: contradicted — The answer gives 45 solo turns or 48 including joint turns instead of the required exact count of 148.
- unsupported: Kosuzu has 45 solo dialogue turns in the entirety of 104 Episode 2.
- unsupported: Kosuzu has 48 total dialogue turns in 104 Episode 2 when joint lines are included.

### a017_omicho_market_history (heuristic)
- sachi: missing — The answer omits Sachi taking the members to help shops, her explanation about local support, and the precedent set by her senior.
- live: partial — It mentions an outdoor performance near the market but omits the small park stage and the market staff audience.
- main: partial — It identifies Tsuzuri bringing Sayaka in 103 Episode 5 without specifying shop assistance, and entirely omits Reika's remarks in 104 Episode 8.
- unsupported: The initial market performance was by DOLLCHESTRA consisting of second-year club president Sachi Ogami and first-year Tsuzuri Yugiri.
- unsupported: The outdoor performance featured Sparkly Spot.
- unsupported: That performance began easing Tsuzuri's longstanding dance insecurities.
- unsupported: Sayaka was facing an expressive plateau when Tsuzuri brought her to the market.
- unsupported: The market experience helped Sayaka stop merely imitating her senior and sister and discover her own authentic sparkle.
- unsupported: Ceras and Izumi were transfer students.
- unsupported: In 105 Episode 4, Kosuzu mentored Hiromi as her part-time coworker at Omicho Market.
- unsupported: Kosuzu helped Hiromi overcome past trauma and stage fright to perform at the Small Triangle School Idol Carnival.
- unsupported: Hiromi subsequently resumed work at Omicho Market and assisted the Bloom Garden Party Coalition.

### a018_ginko_new_tradition (heuristic)
- festival: partial — The answer describes sustaining Bloom Garden Party as a tradition but omits the explicit goal of a new Kanazawa festival lasting years or decades and holding it this school year.

### a019_kaho_sayaka_arc (heuristic)
- start: missing — The answer substitutes early club membership for the bus meeting and omits Sayaka's rejection, warning that knowing her would not be fun, and request to change seats.
- end: partial — The answer correctly identifies their co-presidency but omits Sayaka's proposal that Kaho become president with her support.
- unsupported: Kaho and Sayaka join the club together after being inspired by Kozue and Tsuzuri.
- unsupported: Kaho and Sayaka collaborate to uncover their seniors' past heartbreak and perform together to mend the club.
- unsupported: Their Love Live! victory is specifically a national championship playoff victory.
- unsupported: Kozue, Tsuzuri, and Megumi graduate at the Renge Festival before Kaho and Sayaka take over.

### a021_sayaka_presidency_false_premise (heuristic)
- premise: partial — The answer rejects the skating premise and explains that she requested time, but omits her belief that someone with a big dream should lead.

### a002_reika_san_identity (llm_router)
- market: partial — The answer identifies a familiar local shopkeeper but omits Omicho Market and Tsuzuri bringing Sayaka to help her in 103 Episode 5.
- just_standing: missing — The answer omits both Tsuzuri's earlier standing-only role and Reika's permission for proper work experience in 104 Episode 8.

### a003_kaho_transfer_papers (llm_router)
- papers: missing — The answer discusses brochures but omits her classmates discovering the transfer papers and worrying she will leave.
- stays: missing — The answer explicitly declines to identify the action with the papers and never states her announcement that she will stay.
- change: missing — The answer does not mention her admission of blaming Hasunosora or her vow to change the school herself.
- unsupported: Kaho ultimately decides to join the School Idol Club alongside Sayaka Murano, inspired by Kozue and Tsuzuri.
- abstention mismatch (abstained=True)

### a004_burupa_union (llm_router)
- name: missing — The answer declines to identify the union or provide its full name.
- why: missing — The answer provides none of the required sequence from December Starring Bloom to nationwide parties to the union's formation.
- members: missing — The answer does not describe the members as people inspired by watching Starring Bloom.
- abstention mismatch (abstained=True)

### a006_joint_presidency (llm_router)
- kozue_offer: partial — The answer states that Kozue asked Sayaka to succeed her but omits that Sayaka put the offer on hold.
- sayaka_proposes: missing — The answer omits Sayaka's belief that a leader should have a big dream and her proposal that Kaho become president.
- joint: partial — The answer correctly identifies Kaho and Sayaka as co-presidents but omits Kaho's proposal and Sayaka's agreement.
- unsupported: Kozue overcame post-championship creative burnout before asking Sayaka to succeed her.
- unsupported: Kozue decided to attend a vocal music college before asking Sayaka to succeed her.

### a007_link_to_the_future (llm_router)
- medley: missing — The answer describes a unified competition song but never identifies it as a medley combining three unit songs with an all-member song.
- lost: missing — The answer never states that it is the previous year's song or that they lost with it in the previous Love Live! final.
- nine: partial — The answer identifies a nine-member version but omits the decision to rebuild the existing song and debut it at the Rindou Festival before the qualifiers.
- unsupported: Izumi Katsuragi is a rival scout.

### a008_kosuzu_solo_live (llm_router)
- song: contradicted — The answer rejects the solo-live premise and substitutes an earlier duet rather than identifying the revenge performance of Sparkly Spot before a crowd.
- stumbles: contradicted — The substituted performance is described as mistake-free instead of showing Kosuzu falling, making mistakes, and continuing.
- sayaka: missing — The answer omits Sayaka's reassurance that Kosuzu's feelings reached her, her statement that the live saved her past, and her promise to remain Kosuzu's goal.
- abstention mismatch (abstained=True)

### a009_tsuzuri_job_experiences (llm_router)
- city_hall: partial — The answer correctly identifies city hall clerical work and the dorm mother as arranger, but omits the assigned data-entry task.
- dyeing: partial — The answer correctly credits Ginko and her family's shop connection, but never identifies the traditional dyeing shop or customer service at its hands-on corner.
- market: partial — The answer correctly says Tsuzuri asked Reika-san herself, but does not identify the workplace as Reika-san's market shop.

### a010_kozue_invites_kaho (llm_router)
- manager: missing — The answer abstains without identifying manager as the role Kozue says she offered Kaho.
- idol: missing — The answer never states that Kozue actually wanted to invite Kaho as a school idol too.
- abstention mismatch (abstained=True)

### a014_reika_surname (llm_router)
- no_surname: missing — The answer reports a failed glossary lookup rather than stating that the story never gives Reika's surname.
- abstention mismatch (abstained=True)

### a016_kosuzu_turn_count (llm_router)
- count: contradicted — The answer gives 0 dialogue turns instead of the required exact count of 148.
- unsupported: Kosuzu has 0 dialogue turns in arc 104, episode 2.

### a017_omicho_market_history (llm_router)
- sachi: partial — The answer credits Sachi with local promotional efforts but omits Shades of Stars, helping at shops, the necessity of local support, and her own senior's precedent.
- live: partial — It mentions an outdoor performance near the market but omits the small park stage and market staff audience.
- main: partial — It identifies Tsuzuri bringing Sayaka in 103 Episode 5 but omits their helping and the required 104 Episode 8 return with Reika's assessment of Tsuzuri's growth.
- unsupported: Sachi was club president during the described Year 102 market visit.
- unsupported: DOLLCHESTRA then consisted of Sachi and Tsuzuri and performed Sparkly Spot near Omicho Market.
- unsupported: The market performance eased Tsuzuri's longstanding insecurities about her dancing.
- unsupported: The origin performance is placed in Year 102 Main rather than identified as the Shades of Stars side story.
- unsupported: Tsuzuri took Sayaka to the market specifically to overcome an expressive plateau, where Sayaka discovered her personal sparkle rather than imitating her sister and Tsuzuri.
- unsupported: In Year 105 Episode 4, Kosuzu mentored her part-time coworker Hiromi through severe stage fright, enabling her to perform at the Small Triangle School Idol Carnival and continue working at Omicho Market.

### a018_ginko_new_tradition (llm_router)
- festival: partial — The answer describes Bloom Garden Party as a lasting Kanazawa tradition for years or decades but omits her commitment to hold it this school year.

### a019_kaho_sayaka_arc (llm_router)
- start: partial — The answer describes Sayaka's guarded rejection and request to change seats but omits her warning that getting involved with her would not be fun.
- end: partial — The answer identifies their joint presidency but omits Sayaka's initial offer to support Kaho as president before Kaho proposes sharing the role.
- unsupported: Kaho and Sayaka team up to uncover their seniors' painful history concerning the previous year's Love Live! withdrawal.
- unsupported: They successfully orchestrate a four-person Nadeshiko Festival performance to mend Kozue and Tsuzuri's rift.
- unsupported: They suffer defeat at the national finals in Tokyo and share their grief afterward.
- unsupported: During 104, Kaho and Sayaka become mentors specifically for incoming freshmen Ginko Momose and Kosuzu Kachimachi.
- unsupported: They serve as the central pillars of a nine-member ensemble that wins a national championship playoff.

### a021_sayaka_presidency_false_premise (llm_router)
- premise: partial — The answer rejects the skating premise and describes her postponement of Kozue's offer, but omits her belief that someone with a big dream should lead.
- unsupported: Sayaka and Kaho served together as co-presidents throughout the entire 105th term.

### a002_reika_san_identity (agentic (before fallback fix))
- just_standing: partial — The answer explains Reika-san having Tsuzuri simply stand there but omits her permission for proper work experience in 104 Episode 8.

### a003_kaho_transfer_papers (agentic (before fallback fix))
- no final answer: agent stopped (usage_limit) and returned raw tool output

### a013_unnamed_fan (agentic (before fallback fix))
- no final answer: agent stopped (usage_limit) and returned raw tool output

### a014_reika_surname (agentic (before fallback fix))
- no final answer: agent stopped (usage_limit) and returned raw tool output

### a017_omicho_market_history (agentic (before fallback fix))
- main: partial — Both required main-story returns are identified, but Reika's explicit comparison with Tsuzuri two years earlier is omitted.
- unsupported: The park performance was DOLLCHESTRA's very first mini-live.
- unsupported: Reika-san runs the seafood bowl shop.
- unsupported: Tsuzuri creates artwork reflecting Omicho Market's warm atmosphere during her market visit; the retrieved artwork scene describes a shop without establishing that it is at the market.

### a018_ginko_new_tradition (agentic (before fallback fix))
- festival: partial — The answer describes a lasting communal tradition but omits the explicit goal of making Bloom Garden Party Kanazawa’s new festival and holding it this school year.

### a019_kaho_sayaka_arc (agentic (before fallback fix))
- start: partial — The answer describes Sayaka's coldness and reluctance to make friends on the bus but omits her warning that knowing her would not be fun and her request to change seats.

### a022_tsuzuri_not_a_school_idol (agentic (before fallback fix))
- no final answer: agent stopped (usage_limit) and returned raw tool output

### a002_reika_san_identity (agentic (after fallback fix))
- market: partial — The answer identifies Reika as an Omicho Market worker Tsuzuri helps, but omits Tsuzuri bringing Sayaka to help her in 103 Episode 5.
- just_standing: partial — The answer describes Tsuzuri previously just standing at the shop, but omits Reika allowing her to work properly as a job experience in 104 Episode 8.

### a003_kaho_transfer_papers (agentic (after fallback fix))
- papers: partial — The answer conveys the classmates' concern that she is transferring but omits their earlier discovery of her transfer papers.
- stays: partial — The answer describes gathering her classmates and announcing she will stay, but does not clearly say she does away with the papers.

### a017_omicho_market_history (agentic (after fallback fix))
- main: partial — The answer covers Tsuzuri bringing Sayaka in 103 Episode 5 but omits Reika's comment about Tsuzuri's growth in 104 Episode 8.
- unsupported: The market connection originated with Sachi's upperclassmen as a longstanding club tradition; the passages establish only that her senior had also brought her.
- unsupported: Tsuzuri's distress during the side-story visit was caused by past trauma.
- unsupported: Sachi personally set up the park stage.
- unsupported: Reika-san runs the seafood shops; the retrieved account instead describes her working part-time at different shops.
- unsupported: The retrieved 105th-year self-discovery market scene is specifically Episode 1, Part 4.
- unsupported: The retrieved nodoguro campaign scene is specifically 105 Episode 11, Part 4, and part of a stream named 'Hanasaita-ne Haishin' within 'Triple Flower Daisakusen'.
- unsupported: Kozuzu's market-inspired collaboration suggestion occurs specifically in 105 Episode 6, Part 1.

### a018_ginko_new_tradition (agentic (after fallback fix))
- festival: partial — The answer mentions a lasting festival tradition but omits the Bloom Garden Party goal, its years-or-decades scope, and the plan to hold it this school year.
- tradition: partial — The answer mentions establishing a new lasting tradition but does not specify that it should continue after Ginko is gone.

### a019_kaho_sayaka_arc (agentic (after fallback fix))
- start: missing — The answer describes them as fellow freshmen but omits the bus encounter, Sayaka's rejection of friendship, and her request to change seats.
- unsupported: Kaho and Sayaka join the School Idol Club at the same time and pair with Kozue in Cerise Bouquet and Tsuzuri in DOLLCHESTRA, respectively.
- unsupported: As first-years, Kaho and Sayaka uncover their seniors' heartbreak over a previous Love Live! withdrawal and organize a four-person Nadeshiko Festival stage that mends Kozue and Tsuzuri's rift.
