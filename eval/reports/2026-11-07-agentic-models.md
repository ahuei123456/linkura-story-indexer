# Answer benchmark report

- Suite: `eval/answer_cases.json` version 1
- Generation: varies by column (google:gemini-3.8-flash, openai:gpt-6-luna); answer prompt answer-v3; summary prompt 4
- Judge: varies by column (openai:gpt-6.1-sol, google:gemini-4-argon)
- Corpus: story:01aa9b90dbbf summaries:6376951f185b; code cdde5e6-dirty
- Costs are estimates from configured list prices, not provider invoices. Embedding calls are not metered.

## Summary by mode

| Metric | Flash · Sol judge | Luna · Sol judge | Flash · Argon judge | Luna · Argon judge |
|---|---|---|---|---|
| Cases (judged) | 22 (22) | 22 (22) | 22 (22) | 22 (22) |
| Pass rate | 77% | 27% | 77% | 45% |
| Mean completeness | 92% | 67% | 92% | 78% |
| No contradictions/forbidden claims | 100% | 91% | 100% | 95% |
| No unsupported statements | 91% | 82% | 91% | 95% |
| Abstention appropriate | 100% | 100% | 100% | 100% |
| Required evidence seen | 89% | 68% | 89% | 68% |
| Required evidence in answer context | 89% | 68% | 89% | 68% |
| Citations valid | 100% | 100% | 100% | 100% |
| Temporal leakage | 0% | 0% | 0% | 0% |
| Errors | 0 | 0 | 0 | 0 |
| Agent stopped without answering | 0 | 0 | 0 | 0 |
| Mean latency | 24.5s | 12.1s | 24.5s | 12.1s |
| Model requests | 103 | 95 | 103 | 95 |
| Tokens in/out | 1,411,054/38,967 | 1,279,347/13,057 | 1,411,054/38,967 | 1,279,347/13,057 |
| Est. system cost | $1.2044 | $0.0722 | $1.2044 | $0.0722 |
| Est. judge cost | $0.9383 | $0.8147 | $0.7307 | $0.6315 |

## Evidence vs answer

Whether every required passage reached the answer context, against whether the answer passed.

| Mode | evidence ✓ pass | evidence ✓ fail | evidence ✗ pass | evidence ✗ fail |
|---|---|---|---|---|
| Flash · Sol judge | 14 | 3 | 0 | 2 |
| Luna · Sol judge | 5 | 8 | 0 | 6 |
| Flash · Argon judge | 14 | 3 | 0 | 2 |
| Luna · Argon judge | 7 | 6 | 1 | 5 |

## Pass rate by category

| Category | Flash · Sol judge | Luna · Sol judge | Flash · Argon judge | Luna · Argon judge |
|---|---|---|---|---|
| alias | 80% | 40% | 80% | 80% |
| causal | 62% | 12% | 62% | 25% |
| chronology | 100% | 0% | 100% | 100% |
| count | 100% | 50% | 100% | 50% |
| fact | 100% | 33% | 100% | 33% |
| interleave | 0% | 0% | 0% | 0% |
| quote | 100% | 33% | 100% | 33% |
| relationship_change | 67% | 33% | 67% | 67% |
| unanswerable | 100% | 50% | 100% | 100% |

## Cases

| Case | Flash · Sol judge | Luna · Sol judge | Flash · Argon judge | Luna · Argon judge |
|---|---|---|---|---|
| a001_kaho_sayaka_first_meeting | PASS / ev ✓ | PASS / ev ✓ | PASS / ev ✓ | PASS / ev ✓ |
| a002_reika_san_identity | partial 50% / ev ✓ | partial 25% / ev ✓ | partial 50% / ev ✓ | partial 50% / ev ✓ |
| a003_kaho_transfer_papers | partial 67% / ev ✗ | partial 33% / ev ✗ | partial 83% / ev ✗ | partial 50% / ev ✗ |
| a004_burupa_union | PASS / ev ✓ | partial 83% / ev ✓ | PASS / ev ✓ | PASS / ev ✓ |
| a005_sachi_bgp_role | PASS / ev ✓ | PASS / ev ✓ | PASS / ev ✓ | PASS / ev ✓ |
| a006_joint_presidency | PASS / ev ✓ | PASS / ev ✓ | PASS / ev ✓ | PASS / ev ✓ |
| a007_link_to_the_future | PASS / ev ✓ | partial 50% / ev ✓ | PASS / ev ✓ | partial 50% / ev ✓ |
| a008_kosuzu_solo_live | PASS / ev ✓ | partial 67% / ev ✓ | PASS / ev ✓ | PASS / ev ✓ |
| a009_tsuzuri_job_experiences | PASS / ev ✓ | WRONG / ev ✗ | PASS / ev ✓ | partial 83% / ev ✗ |
| a010_kozue_invites_kaho | PASS / ev ✓ | partial 75% / ev ✗ | PASS / ev ✓ | partial 75% / ev ✗ |
| a011_third_year_career_plans | PASS / ev ✓ | partial 67% / ev ✗ | PASS / ev ✓ | partial 83% / ev ✗ |
| a012_tsuzuri_advice_quote | PASS / ev ✓ | PASS / ev ✓ | PASS / ev ✓ | PASS / ev ✓ |
| a013_unnamed_fan | PASS / ev ✓ | PASS / ev ✓ | PASS / ev ✓ | PASS / ev ✓ |
| a014_reika_surname | PASS | partial 50% | PASS | PASS |
| a015_reika_line_count | PASS | PASS | PASS | PASS |
| a016_kosuzu_turn_count | PASS | WRONG | PASS | WRONG |
| a017_omicho_market_history | partial 83% / ev ✓ | partial 50% / ev ✓ | partial 83% / ev ✓ | partial 50% / ev ✓ |
| a018_ginko_new_tradition | partial 67% / ev ✓ | partial 67% / ev ✓ | partial 50% / ev ✓ | partial 67% / ev ✓ |
| a019_kaho_sayaka_arc | partial 50% / ev ✗ | partial 50% / ev ✓ | partial 50% / ev ✗ | partial 75% / ev ✓ |
| a020_episode1_order | PASS / ev ✓ | partial 83% / ev ✗ | PASS / ev ✓ | PASS / ev ✗ |
| a021_sayaka_presidency_false_premise | PASS / ev ✓ | partial 75% / ev ✗ | PASS / ev ✓ | partial 75% / ev ✗ |
| a022_tsuzuri_not_a_school_idol | PASS / ev ✓ | partial 50% / ev ✓ | PASS / ev ✓ | partial 50% / ev ✓ |

## Failure notes

### a002_reika_san_identity (Flash · Sol judge)
- market: partial — The answer identifies Reika as an Omicho Market worker Tsuzuri helps, but omits Tsuzuri bringing Sayaka to help her in 103 Episode 5.
- just_standing: partial — The answer describes Tsuzuri previously just standing at the shop, but omits Reika allowing her to work properly as a job experience in 104 Episode 8.

### a003_kaho_transfer_papers (Flash · Sol judge)
- papers: partial — The answer conveys the classmates' concern that she is transferring but omits their earlier discovery of her transfer papers.
- stays: partial — The answer describes gathering her classmates and announcing she will stay, but does not clearly say she does away with the papers.

### a017_omicho_market_history (Flash · Sol judge)
- main: partial — The answer covers Tsuzuri bringing Sayaka in 103 Episode 5 but omits Reika's comment about Tsuzuri's growth in 104 Episode 8.
- unsupported: The market connection originated with Sachi's upperclassmen as a longstanding club tradition; the passages establish only that her senior had also brought her.
- unsupported: Tsuzuri's distress during the side-story visit was caused by past trauma.
- unsupported: Sachi personally set up the park stage.
- unsupported: Reika-san runs the seafood shops; the retrieved account instead describes her working part-time at different shops.
- unsupported: The retrieved 105th-year self-discovery market scene is specifically Episode 1, Part 4.
- unsupported: The retrieved nodoguro campaign scene is specifically 105 Episode 11, Part 4, and part of a stream named 'Hanasaita-ne Haishin' within 'Triple Flower Daisakusen'.
- unsupported: Kozuzu's market-inspired collaboration suggestion occurs specifically in 105 Episode 6, Part 1.

### a018_ginko_new_tradition (Flash · Sol judge)
- festival: partial — The answer mentions a lasting festival tradition but omits the Bloom Garden Party goal, its years-or-decades scope, and the plan to hold it this school year.
- tradition: partial — The answer mentions establishing a new lasting tradition but does not specify that it should continue after Ginko is gone.

### a019_kaho_sayaka_arc (Flash · Sol judge)
- start: missing — The answer describes them as fellow freshmen but omits the bus encounter, Sayaka's rejection of friendship, and her request to change seats.
- unsupported: Kaho and Sayaka join the School Idol Club at the same time and pair with Kozue in Cerise Bouquet and Tsuzuri in DOLLCHESTRA, respectively.
- unsupported: As first-years, Kaho and Sayaka uncover their seniors' heartbreak over a previous Love Live! withdrawal and organize a four-person Nadeshiko Festival stage that mends Kozue and Tsuzuri's rift.

### a002_reika_san_identity (Luna · Sol judge)
- market: partial — The answer places Reika in a shopping district and mentions Tsuzuri bringing Sayaka, but does not identify Omicho Market or explain that they come to help her.
- just_standing: missing — The answer omits both Tsuzuri's former standing-only role and Reika's permission for proper work experience in 104 Episode 8.

### a003_kaho_transfer_papers (Luna · Sol judge)
- papers: missing — The answer does not mention her classmates discovering the transfer papers or worrying that she is leaving.
- stays: partial — It states that she tears up the papers and decides to stay, but omits gathering her classmates and announcing her decision to them.
- change: partial — It conveys her resolve to make the school more fun but omits her admission that she had blamed Hasunosora for everything.

### a004_burupa_union (Luna · Sol judge)
- why: partial — It explains the nationwide series and the club's inability to run it alone, but omits December and does not explicitly establish that the nationwide decision followed Starring Bloom before the union formed.

### a007_link_to_the_future (Luna · Sol judge)
- medley: partial — The answer identifies a medley combining the three units' strengths but omits the all-member song component.
- lost: missing — The answer never states that it is the previous year's song or that they lost with it in the Love Live! final.

### a008_kosuzu_solo_live (Luna · Sol judge)
- song: partial — The answer identifies Sparkly Spot, DOLLCHESTRA, and the crowd, but omits Kosuzu's framing of the live as revenge.
- sayaka: partial — The answer captures Sayaka's reassessment of her childhood and promise to remain Kosuzu's goal, but omits her explicit reassurance that the live was not a failure because Kosuzu's feelings reached her.

### a009_tsuzuri_job_experiences (Luna · Sol judge)
- dyeing: partial — The answer correctly identifies the dyeing workshop, family connection, customer service, and Ginko's arrangement but omits the hands-on experience corner.
- market: contradicted — The answer attributes arrangement to Reika-san rather than Tsuzuri herself asking Reika-san.
- unsupported: The market placement was arranged by Reika-san.

### a010_kozue_invites_kaho (Luna · Sol judge)
- idol: partial — The answer says Kozue wanted Kaho to join but does not specify that she wanted to invite her as a school idol too.
- unsupported: The answer attributes the revelation to 'Tsurugi,' whereas the retrieved passage attributes it to 綴理 (Tsuzuri).

### a011_third_year_career_plans (Luna · Sol judge)
- sayaka: partial — The answer gives university and continued figure skating but omits applying her school idol experience to skating.
- kaho: partial — The answer states her university plan and many interests but does not explicitly convey that she cannot yet specify what she wants to do.

### a014_reika_surname (Luna · Sol judge)
- no_surname: partial — The answer says no fuller name appears in the retrieved scene but does not establish that the story never gives her surname.
- abstention mismatch (abstained=True)

### a016_kosuzu_turn_count (Luna · Sol judge)
- count: contradicted — The answer states 0 dialogue turns rather than the required exact count of 148.

### a017_omicho_market_history (Luna · Sol judge)
- live: missing — The answer mentions a performance goal but not the actual live on a small park stage near the market for market staff.
- main: partial — The answer includes Tsuzuri bringing Sayaka in Episode 5 but omits Reika's reflection on Tsuzuri's growth in 104 Episode 8.
- unsupported: The Kaho livestream and nodoguro-overstock incident occurs in Episode 11.

### a018_ginko_new_tradition (Luna · Sol judge)
- festival: partial — The answer describes making Bloom Garden Party a lasting Kanazawa festival but omits the goal of holding it within the current school year.
- tradition: partial — The answer describes a shared enduring tradition but does not state that it should continue after Ginko is gone.

### a019_kaho_sayaka_arc (Luna · Sol judge)
- start: partial — The answer captures Sayaka's initial reserve and rejection of closeness but omits the bus setting, her warning that knowing her will not be fun, and her request to change seats.
- end: partial — The answer states their agreement to lead together but omits Sayaka's initial proposal that Kaho become president with Sayaka supporting her.

### a020_episode1_order (Luna · Sol judge)
- third: partial — The answer correctly places her decision to stay last but omits her promise to change the school.

### a021_sayaka_presidency_false_premise (Luna · Sol judge)
- premise: partial — The answer rejects the skating premise and notes that Sayaka deferred Kozue's offer, but omits her belief that someone with a big dream should lead.
- unsupported: Sayaka realizes she had been driven by wanting to be needed.

### a022_tsuzuri_not_a_school_idol (Luna · Sol judge)
- contest: missing — The answer describes where she recounts the remark but never says she originally heard it at a school idol competition.

### a002_reika_san_identity (Flash · Argon judge)
- market: partial — The answer identifies Reika-san as an Omicho Market shop worker whom Tsuzuri helps out, though it does not explicitly mention Tsuzuri bringing Sayaka in 103 Episode 5.
- just_standing: partial — The answer mentions that Tsuzuri previously just stood at the storefront, but omits that in 104 Episode 8 Reika-san lets her properly work there as one of her job experiences.

### a003_kaho_transfer_papers (Flash · Argon judge)
- papers: partial — The answer mentions Biwako asking if Kaho called them out to say goodbye because she is transferring, but does not mention the earlier scene where her classmates catch her with the transfer papers (転入届).

### a017_omicho_market_history (Flash · Argon judge)
- main: partial — The answer covers Tsuzuri bringing Sayaka to help at the market in 103 Episode 5, as well as several 105th-year main story scenes, but omits the 104 Episode 8 scene where Reika-san remarks on how much Tsuzuri has changed since two years earlier.
- unsupported: Tsuzuri brings her first-year junior, Sayaka Murano, to Omicho Market at 5:00 AM

### a018_ginko_new_tradition (Flash · Argon judge)
- festival: missing — The answer mentions continuing the festival as a lasting tradition, but completely omits the Bloom Garden Party by name and the goal of holding it this school year as a new Kanazawa festival lasting years or decades.
- tradition: partial — The answer mentions establishing a new, lasting tradition, though it leaves out the detail about it continuing after she is gone.

### a019_kaho_sayaka_arc (Flash · Argon judge)
- start: missing — The answer does not describe their first meeting on the bus or Sayaka's initial standoffishness and request to change seats.

### a002_reika_san_identity (Luna · Argon judge)
- just_standing: missing — The answer does not mention that Tsuzuri used to just stand at the shop or that she later worked there properly for her job experience.

### a003_kaho_transfer_papers (Luna · Argon judge)
- papers: missing — The answer does not mention the earlier scene where her classmates catch her with the transfer papers.
- change: partial — The answer states that she decides to make the school more fun for everyone, though it omits her admission that she had been blaming Hasunosora for everything.

### a007_link_to_the_future (Luna · Argon judge)
- medley: partial — The answer states it is a medley bringing together the strengths of the three units, though it omits mentioning the all-member song component.
- lost: missing — The answer does not mention that it is the song from the previous year's Love Live! final or that they lost with it.

### a009_tsuzuri_job_experiences (Luna · Argon judge)
- market: partial — The answer mentions working at the local market involving Reika-san, saying it was arranged by Reika-san rather than specifying that Tsuzuri asked Reika-san herself to work at her shop.

### a010_kozue_invites_kaho (Luna · Argon judge)
- idol: partial — The answer only says Kozue wanted Kaho to "join the club" (based on Tsurugi's remark) rather than stating that Kozue herself admits she actually wanted to invite Kaho as a school idol too.

### a011_third_year_career_plans (Luna · Argon judge)
- kaho: partial — The answer states Kaho will go to university because there are many things she wants to do, which implies she hasn't narrowed it down yet, though it omits explicitly stating she cannot yet say what specifically she wants to do.

### a016_kosuzu_turn_count (Luna · Argon judge)
- count: contradicted — The answer claims Kosuzu has 0 dialogue turns, contradicting the required count of 148.
- unsupported: Kosuzu has 0 dialogue turns in Episode 2 of arc 104.

### a017_omicho_market_history (Luna · Argon judge)
- live: missing — The answer does not mention the live performance held on the small stage in a park near the market for the market staff.
- main: partial — The answer covers Tsuzuri bringing Sayaka to help at the market in Episode 5 (103), but omits the 104 Episode 8 return where Reika-san remarks on how much Tsuzuri has changed since two years earlier (instead citing a different Episode 11 event).

### a018_ginko_new_tradition (Luna · Argon judge)
- festival: partial — The answer states she wants to turn Bloom Days into the Bloom Garden Party as a festival that continues for years, though it omits holding it this school year.
- tradition: partial — The answer mentions creating a shared tradition that everyone helps sustain, though it does not explicitly mention it continuing after she is gone.

### a019_kaho_sayaka_arc (Luna · Argon judge)
- start: partial — The answer mentions that Sayaka is initially reserved and explicitly says she does not intend to get close to people, though it omits the specific details of the bus and asking to change seats.

### a021_sayaka_presidency_false_premise (Luna · Argon judge)
- premise: partial — The answer rejects the premise that she turned it down for skating and notes she asked Kozue for time to think, but it omits her actual reason for putting the offer on hold (that someone with a big dream should lead the club).

### a022_tsuzuri_not_a_school_idol (Luna · Argon judge)
- contest: missing — The answer interprets "where" as the episode/scene where Tsuzuri recounts the memory rather than stating that she was told this at a school idol competition.
