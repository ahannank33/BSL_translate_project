# BSL Alphabet Reference Guide

This guide shows the hand shapes for each letter in the British Sign Language (BSL) alphabet. Use this as reference when collecting data.

## Important Notes

**BSL vs ASL:**
- BSL (British) uses **TWO-handed** signs for many letters
- ASL (American) uses **ONE-handed** signs
- Make sure you're using BSL specifically!

**For Data Collection:**
- Some BSL letters require both hands (C, G, H, P, Q)
- Most BSL letters use one hand
- Keep your hand steady and well-lit
- Ensure fingers are clearly visible

## BSL Alphabet Quick Reference

### One-Handed Letters

**A** - Closed fist with thumb alongside

**B** - Flat hand, fingers together, thumb across palm

**D** - Index finger pointing up, thumb and middle finger form circle

**E** - Bent fingers, thumb across

**F** - Thumb and index finger form circle, other fingers extended

**I** - Pinky finger extended, other fingers closed, thumb across

**J** - Like I, but with a J-shaped motion

**K** - Index and middle finger up in V-shape, thumb between them

**L** - Thumb and index finger form L-shape

**M** - Thumb under three fingers

**N** - Thumb under two fingers

**O** - Fingertips and thumb form O-shape

**R** - Index and middle fingers crossed

**S** - Closed fist, thumb across fingers

**T** - Thumb between index and middle finger

**U** - Index and middle fingers together, pointing up

**V** - Index and middle fingers in V-shape (like peace sign)

**W** - Three fingers up (index, middle, ring)

**X** - Index finger bent into hook shape

**Y** - Thumb and pinky extended (hang loose gesture)

**Z** - Make Z-shape in the air with index finger

### Two-Handed Letters

**C** - Both hands form C-shapes

**G** - Both hands, index fingers pointing at each other

**H** - Both hands horizontal, one above the other

**P** - Both hands involved in formation

**Q** - Both hands form Q-shape

## Tips for Data Collection

1. **Consistent Positioning**: Keep your hand at a similar distance from camera for all captures
2. **Good Lighting**: Ensure your hand is well-lit from the front
3. **Neutral Background**: Plain backgrounds help the model focus on hand shape
4. **Slight Variations**: For each letter, vary your hand position slightly (small rotations, different positions in frame)
5. **Clear Gestures**: Make the sign clearly - don't rush between letters

## Data Collection Strategy

For best results, collect in batches:

1. **Batch 1** (Days 1-2): Collect all one-handed letters (A, B, D, E, F, I, J, K, L, M, N, O, R, S, T, U, V, W, X, Y, Z)
2. **Batch 2** (Day 3): Collect two-handed letters (C, G, H, P, Q)
3. **Review** (Day 4): Test collected data, fill in any letters with low sample counts

**Target: 500 samples per letter = 13,000 total samples**

## Resources

For visual reference of BSL alphabet:
- British Deaf Association: https://bda.org.uk/
- Search YouTube for "BSL Alphabet Tutorial"
- BSL SignBank: https://bslsignbank.ucl.ac.uk/

## Note on Two-Handed Signs

The current `data_collection.py` is set to detect only 1 hand. For two-handed letters (C, G, H, P, Q), you'll need to:

1. Either modify the script to detect 2 hands (change `max_num_hands=2`)
2. Or start with one-handed letters only and add two-handed letters later
3. Or use both hands but ensure the primary hand is clearly visible

**Recommendation for Phase 1**: Start with one-handed letters only (21 letters) to build a working system faster. Add two-handed letters in Phase 2.
