"""Write data/labeled_eval.csv — 100 hand-labeled reviews for evaluation.

Each row was labeled by hand (Freddy Thosago, Oct 2026), including edge
cases: sarcasm, ALL CAPS, emojis, mixed sentiment, slang, and reviews with
no aspect (unassigned). Usage: python data/make_labeled_eval.py
"""
from __future__ import annotations

from pathlib import Path

# (review_id, text, gold_overall, gold_aspects)
EVAL = [
 ("e001","The courier delivered a day early and the packaging was sealed properly. Great service.","Positive","delivery:Positive;packaging:Positive"),
 ("e002","Delivery was two weeks late and nobody at the call centre could tell me where my parcel was.","Negative","delivery:Negative;customer service:Negative"),
 ("e003","Excellent value for money, the quality is sturdy and it feels well made.","Positive","price:Positive;quality:Positive"),
 ("e004","The price is a rip-off for what you get. Broke after two days.","Negative","price:Negative;quality:Negative"),
 ("e005","Customer service never replied to my WhatsApp messages about my refund.","Negative","customer service:Negative;returns & refunds:Neutral"),
 ("e006","The colour looks nothing like the picture. Not as described at all.","Negative","product accuracy:Negative"),
 ("e007","Lekker product, works like a charm.","Positive",""),
 ("e008","It is okay I suppose. Delivery took about a week.","Neutral","delivery:Neutral"),
 ("e009","The box arrived crushed and the kettle inside was dented.","Negative","packaging:Negative;quality:Negative"),
 ("e010","Refund was processed in two days, painless exchange.","Positive","returns & refunds:Positive"),
 ("e011","Great quality but the shipping cost more than the item itself.","Neutral","quality:Positive;delivery:Negative"),
 ("e012","Support agent was friendly and sorted my return quickly.","Positive","customer service:Positive;returns & refunds:Positive"),
 ("e013","Still waiting on my parcel after a month. Eish.","Negative","delivery:Negative"),
 ("e014","Genuine product, exactly as described, arrived fast.","Positive","product accuracy:Positive;delivery:Positive"),
 ("e015","Received a fake. Definitely a replica of the real thing.","Negative","product accuracy:Negative"),
 ("e016","The price is average for this kind of product.","Neutral","price:Neutral"),
 ("e017","Battery lasts the whole day, love it.","Positive","quality:Positive"),
 ("e018","The call centre kept me on hold for 40 minutes and then cut me off.","Negative","customer service:Negative"),
 ("e019","Packaging was a plain box, nothing special but it did the job.","Neutral","packaging:Neutral"),
 ("e020","They refused my return even though the item was faulty.","Negative","returns & refunds:Negative;quality:Negative"),
 ("e021","Fast dispatch and the tracking actually worked.","Positive","delivery:Positive"),
 ("e022","Hmm, the size is more or less as described. Fine I guess.","Neutral","product accuracy:Neutral"),
 ("e023","Cheap material, flimsy build, would not recommend.","Negative","quality:Negative;price:Negative"),
 ("e024","Ordered on Monday, arrived Tuesday. Courier guy is a legend.","Positive","delivery:Positive"),
 ("e025","The refund is still not in my account three weeks later.","Negative","returns & refunds:Negative"),
 ("e026","Beautiful hoodie, true to the photo, and warm.","Positive","product accuracy:Positive"),
 ("e027","Shipping was quick but the box was open when it got here.","Neutral","delivery:Positive;packaging:Negative"),
 ("e028","No bubble wrap, just a flimsy envelope. Luckily nothing broke.","Neutral","packaging:Negative;quality:Positive"),
 ("e029","Worth every rand, honestly the best buy of the year.","Positive","price:Positive"),
 ("e030","Customer service replied within an hour on WhatsApp.","Positive","customer service:Positive"),
 ("e031","Late delivery ruined the birthday gift surprise.","Negative","delivery:Negative"),
 ("e032","Quality is average for this price bracket, nothing fancy.","Neutral","quality:Neutral;price:Neutral"),
 ("e033","The replacement arrived quickly after my first one was defective.","Neutral","quality:Negative;returns & refunds:Positive;delivery:Positive"),
 ("e034","Amazing value and the material feels expensive.","Positive","price:Positive;quality:Positive"),
 ("e035","Emailed support twice about the missing charger, no response.","Negative","customer service:Negative;quality:Negative"),
 ("e036","Power bank stopped working within a week. Very disappointed.","Negative","quality:Negative"),
 ("e037","Packaging was sealed and secure, no complaints.","Positive","packaging:Positive"),
 ("e038","The advert said 64GB but I received the 32GB version.","Negative","product accuracy:Negative"),
 ("e039","Delivery took about five days, which is normal I suppose.","Neutral","delivery:Neutral"),
 ("e040","The agent on the helpline was patient and walked me through the return.","Positive","customer service:Positive;returns & refunds:Positive"),
 ("e041","Way overpriced for a basic office chair.","Negative","price:Negative"),
 ("e042","Exactly what I ordered, correct colour and size.","Positive","product accuracy:Positive"),
 ("e043","The courier never pitched and nobody called back.","Negative","delivery:Negative;customer service:Negative"),
 ("e044","Return process is still ongoing, apparently the refund is approved.","Neutral","returns & refunds:Neutral"),
 ("e045","Sturdy desk lamp, good build quality for the money.","Positive","quality:Positive;price:Positive"),
 ("e046","THE DELIVERY WAS SO LATE AND THE TRACKING NEVER UPDATED. SHOCKING.","Negative","delivery:Negative"),
 ("e047","Charger works perfectly, great little buy 👍","Positive","quality:Positive;price:Positive"),
 ("e048","The sneakers look cheap in person, nothing like the photos.","Negative","product accuracy:Negative;quality:Negative"),
 ("e049","Fair price, decent quality, arrived on time. All good.","Positive","price:Positive;quality:Positive;delivery:Positive"),
 ("e050","Asked for a refund on the fake perfume and they just ignore me now.","Negative","product accuracy:Negative;returns & refunds:Negative;customer service:Negative"),
 ("e051","The yoga mat is fine but the colour is slightly off from the picture.","Neutral","quality:Positive;product accuracy:Negative"),
 ("e052","Thanks to Thandi at support, my exchange was sorted the same day.","Positive","customer service:Positive;returns & refunds:Positive"),
 ("e053","Coffee machine arrived dented because there was no bubble wrap.","Negative","quality:Negative;packaging:Negative"),
 ("e054","Two weeks later, still waiting. Not even an apology email.","Negative","delivery:Negative;customer service:Negative"),
 ("e055","The hair dryer is genuine and the price beat every other store.","Positive","product accuracy:Positive;price:Positive"),
 ("e056","Bag was well wrapped and the box was in perfect shape.","Positive","packaging:Positive"),
 ("e057","Hmm, the warranty question is still unanswered but delivery was fast.","Neutral","customer service:Negative;delivery:Positive"),
 ("e058","They exchanged the faulty speaker without any drama. Respect.","Positive","returns & refunds:Positive;quality:Negative"),
 ("e059","Standard delivery, standard product, standard price. Meh.","Neutral","delivery:Neutral;product accuracy:Neutral;price:Neutral"),
 ("e060","The courier left my parcel at the gate in the rain. Box was soaked.","Negative","delivery:Negative;packaging:Negative"),
 ("e061","Best pricing I found and the quality is honestly surprising.","Positive","price:Positive;quality:Positive"),
 ("e062","Sent three emails about the refund, zero replies.","Negative","returns & refunds:Neutral;customer service:Negative"),
 ("e063","Bluetooth speaker is exactly as advertised and sounds great.","Positive","product accuracy:Positive;quality:Positive"),
 ("e064","The description said stainless steel but this is clearly plastic.","Negative","product accuracy:Negative;quality:Negative"),
 ("e065","Quick dispatch, arrived before the estimated date.","Positive","delivery:Positive"),
 ("e066","Call centre queue music is the only thing they do well. 45 minutes on hold.","Negative","customer service:Negative"),
 ("e067","Decent quality, though the price could be a bit lower honestly.","Neutral","quality:Positive;price:Negative"),
 ("e068","The backpack straps broke on day two. Flimsy stitching.","Negative","quality:Negative"),
 ("e069","Excellent! Fast shipping, sealed box, genuine product.","Positive","delivery:Positive;packaging:Positive;product accuracy:Positive"),
 ("e070","Refund finally came through after a month of chasing.","Neutral","returns & refunds:Negative"),
 ("e071","Size chart is wrong, ordered my usual and it is two sizes small.","Negative","product accuracy:Negative"),
 ("e072","Support was useless, kept sending me the same link.","Negative","customer service:Negative"),
 ("e073","Packaging was ridiculous, three boxes inside boxes, but item is perfect.","Neutral","packaging:Negative;quality:Positive"),
 ("e074","Reasonable shipping fee and the courier was on time.","Positive","delivery:Positive;price:Positive"),
 ("e075","The kettle leaks from the bottom. Dangerous and cheap.","Negative","quality:Negative;price:Negative"),
 ("e076","Photo shows a wooden shelf, I received chipboard. False advertising.","Negative","product accuracy:Negative"),
 ("e077","Service with a smile, they even helped me choose the right size.","Positive","customer service:Positive;product accuracy:Positive"),
 ("e078","Waiting since the 3rd for delivery that was due in 48 hours.","Negative","delivery:Negative"),
 ("e079","For the price, the material should be much better quality.","Negative","price:Negative;quality:Negative"),
 ("e080","Money back in my account two days after the return. Impressive.","Positive","returns & refunds:Positive"),
 ("e081","The hoodie colour in the picture is navy but the actual one is purple.","Negative","product accuracy:Negative"),
 ("e082","Delivery was fine, product is fine, all round okay experience.","Neutral","delivery:Neutral"),
 ("e083","Wow, late AGAIN. Third order in a row with slow shipping.","Negative","delivery:Negative"),
 ("e084","The exchange policy is straightforward, no hidden costs.","Positive","returns & refunds:Positive"),
 ("e085","Great product but the response time from support is terrible.","Neutral","quality:Positive;customer service:Negative"),
 ("e086","Speaker arrived with a cracked casing, packaging was paper thin.","Negative","packaging:Negative;quality:Negative"),
 ("e087","Value for money is unmatched, my friends keep asking where I bought it.","Positive","price:Positive"),
 ("e088","Nobody answers the hotline. Ever.","Negative","customer service:Negative"),
 ("e089","Parcel arrived, tracking updated the whole way, nice touch.","Positive","delivery:Positive"),
 ("e090","The replica watch looks good but I knew it was not the real thing.","Neutral","product accuracy:Neutral;quality:Neutral"),
 ("e091","Second order from them, quality is consistently good.","Positive","quality:Positive"),
 ("e092","They charged me for express shipping and it still took nine days.","Negative","delivery:Negative;price:Negative"),
 ("e093","Chat bot kept me in a loop, then the line dropped.","Negative","customer service:Negative"),
 ("e094","Perfect fit, perfect colour, exactly like the listing.","Positive","product accuracy:Positive"),
 ("e095","Still no refund. Told me five working days, it has been twenty.","Negative","returns & refunds:Negative"),
 ("e096","Affordable and cheerful, delivery was also quicker than promised.","Positive","price:Positive;delivery:Positive"),
 ("e097","The box was soaked but the shoes inside were luckily fine.","Neutral","packaging:Negative;quality:Positive"),
 ("e098","Return label took a week to arrive, but the refund itself was quick.","Neutral","returns & refunds:Neutral"),
 ("e099","Sarcasm test: just LOVE waiting three weeks for a two-day delivery.","Negative","delivery:Negative"),
 ("e100","No aspect here at all, just saying thanks for a smooth experience overall.","Positive",""),
]


def main() -> None:
    """Write the labeled CSV next to this script."""
    lines = ["review_id,text,gold_overall,gold_aspects"]
    for rid, text, overall, aspects in EVAL:
        text = text.replace('"', "'")
        lines.append(f'{rid},"{text}",{overall},"{aspects}"')
    out = Path(__file__).parent / "labeled_eval.csv"
    out.write_text("\n".join(lines) + "\n")
    print(f"Wrote {len(EVAL)} labeled reviews to {out}")


if __name__ == "__main__":
    main()
