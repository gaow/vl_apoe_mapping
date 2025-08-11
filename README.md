# vl_apoe_mapping

## Prompts (to Claude chatbox)

1. Read the Virtual Lab paper attached. I want to setup a virtual lab to study my oen problem: specifically i have APOE region which is very complicated in terms of density of genes and linkage disequilibrium. I also have GWAS for APOE for Alzheimers disease summary statisics and a bunch of molecular QTL data fine-mapped from individual level ata for it. My goal is to find independent signals apart from E2/E3/E4

2. What you've proposed are good but a bit too basic. I've looked into some basic aspects of the problem already, but the challegs are:
    1. APOE signal too strong for E4 it overshadows others. If i condition on E4 unless i have perfoectly matching LD data I cannot get correctly for other loci because they will be overestimated of teh effects
    2. I have various xQTL data indeed in terms of credible sets and they match / overlap with APOE many variants there. but that colocalization does not mean it is correct because again too many signals in APOE that might be due to LD mismatch

3. let's set up the virtual lab for this problem without using any coding.
    - [See the results here](vl_no_code_20250805.md); [initial version](https://github.com/gaow/vl_apoe_mapping/blob/b4d1b333a0eddea83289f361654274a6398c8506/vl_no_code_20250805.md).

4. This reads great now let's use some programmatic way for this to happen, like using some API? How does the Virtual Lab paper do this we will do the same.
    - Then after paying $5 to get the API key
    - It then provides the key codes `simple_setup.py` and `advanced_lab.py` but barely showed how to use them

5. You provded two scripts. It is not clear which script is goingto. be used at what point. The 2nd one seems like the virtual lab setupb but how to use the first one? you need to give very detailed explanations
    - It then provides everthing that works out of the box, generating the output [shown in this commit](https://github.com/gaow/vl_apoe_mapping/tree/bde3370de4b0eacc5aeca54b1ffbde917793a529).
    - Session now reached maximum token limit

## First round of attempts

```bash
python simple_setup.py
python task0_orientation.py
#python task1_xqtl_fp.py
python task2_parallel_meeting.py
```

