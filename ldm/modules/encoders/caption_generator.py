import os
from pathlib import Path
import csv
import sys
import traceback
import argparse
import re
import random

import pandas as pd
from tqdm import tqdm
import numpy as np
import music21


# def load_samples_from_tsv(tsv_path):  # 从tsv中读取文件
#     tsv_path = Path(tsv_path)
#     if not tsv_path.is_file():
#         raise FileNotFoundError(f"Dataset not found: {tsv_path}")
#     with open(tsv_path) as f:
#         reader = csv.DictReader(
#             f,
#             delimiter="\t",
#             quotechar=None,
#             doublequote=False,
#             lineterminator="\n",
#             quoting=csv.QUOTE_NONE,
#         )
#         samples = [dict(e) for e in reader]
#     if len(samples) == 0:
#         raise ValueError(f"Empty manifest: {tsv_path}")
#     return samples
#
# def save_df_to_tsv(dataframe, path):
#     _path = path if isinstance(path, str) else path.as_posix()
#     dataframe.to_csv(
#         _path,
#         sep="\t",
#         header=True,
#         index=False,
#         encoding='utf-8',
#         escapechar='\\',
#         quoting=csv.QUOTE_NONE,
#     )
#
# def handle_exception(err, skipped_name=''):
#     _, exc_value, exc_tb = sys.exc_info()
#     tb = traceback.extract_tb(exc_tb)[-1]
#     if skipped_name != '':
#         print(f'skip {skipped_name}, {err}: {exc_value} in {tb[0]}:{tb[1]} "{tb[2]}" in {tb[3]}')
#     else:
#         print(f'{err}: {exc_value} in {tb[0]}:{tb[1]} "{tb[2]}" in {tb[3]}')


class CaptionGenerator:
    def __init__(self):
        self.key_kw = '[Key]'
        self.avg_pitch_kw = '[pitch level]'
        self.tempo_kw = '[tempo]'
        self.emotion_kw = '[emotional characteristics]'
        self.duration_kw = '[duration]'

        self.key_min_conf = 0.5
        self.tempo_min_conf = 0.3

        # 我现在需要把多个关于一段音乐的旋律性质（调，平均音高，速度，情绪特征，时长）来构造一段对这段旋律的自然语言的描述，比如：“这首歌的片段的旋律是A大调的，音高较高，速度中等，时长较短。旋律中充满了浪漫的情绪氛围”。我现在需要50个可以直接带入替换的这样的自然语言的英文句子模板，句子的结构灵活多变，但是特征必须包含。
        self.full_factor_templates_w_duration = [
            "This melody, set in [Key] with a [pitch level] pitch and [tempo] tempo, lasts [duration] and carries a [emotional characteristics] aura.",
            "In the key of [Key], this segment features a melody with a [pitch level] pitch, a [tempo] tempo, spans [duration], and exudes a [emotional characteristics] atmosphere.",
            "The song’s melody, composed in [Key], holds a [pitch level] pitch and maintains a [tempo] pace, covering [duration], filled with [emotional characteristics] qualities.",
            "With its [pitch level] pitch and [tempo] tempo over [duration], the melody in [Key] radiates [emotional characteristics] emotions.",
            "This musical piece in [Key] resonates with a [pitch level] pitch at a [tempo] tempo for [duration], infused with [emotional characteristics] tone.",
            "The melody of this segment, crafted in [Key] with a [pitch level] pitch and a [tempo] tempo, lasting [duration], captures a [emotional characteristics] mood.",
            "Set in the tonal framework of [Key], this melody, having a [pitch level] pitch and a [tempo] pace, extends for [duration] and projects [emotional characteristics] feel.",
            "The melody, originating in [Key] at a [pitch level] pitch and moving at a [tempo] tempo for [duration], pulses with [emotional characteristics] energy.",
            "This segment’s melody in [Key], with its [pitch level] pitch and [tempo] tempo across [duration], breathes a [emotional characteristics] essence.",
            "In [Key], this melody with a [pitch level] pitch and [tempo] tempo over [duration], is steeped in [emotional characteristics] essence.",
            "The tune, rooted in [Key] and exhibiting a [pitch level] pitch with a [tempo] tempo for [duration], vibrates with [emotional characteristics] qualities.",
            "This melody, from the key of [Key], carrying a [pitch level] pitch and progressing at a [tempo] pace for [duration], is laden with [emotional characteristics] essence.",
            "With a [pitch level] pitch and [tempo] tempo, this melody from [Key], lasting [duration], channels [emotional characteristics] emotions.",
            "The song, played in [Key], maintains a [pitch level] pitch and [tempo] tempo through [duration], resonating with [emotional characteristics] mood.",
            "This piece’s melody, in [Key] at a [pitch level] pitch and a [tempo] tempo over [duration], manifests a [emotional characteristics] ambiance.",
            "The melody, echoing through [Key] with a [pitch level] pitch and [tempo] tempo for [duration], is filled with [emotional characteristics] feel.",
            "Set in [Key], the melody’s [pitch level] pitch and [tempo] tempo, spanning [duration], evoke [emotional characteristics] feels.",
            "In [Key], this song's segment offers a melody with a [pitch level] pitch at a [tempo] pace for [duration], imbued with [emotional characteristics] energy.",
            "This segment, orchestrated in [Key], utilizes a melody with a [pitch level] pitch and a [tempo] tempo lasting [duration], enriched by [emotional characteristics] emotions.",
            "Composed in [Key], the melody upholds a [pitch level] pitch and a [tempo] pace through [duration], suffused with [emotional characteristics] feel.",
            "The melody, unfolding in [Key] with a [pitch level] pitch and a [tempo] tempo for [duration], showcases [emotional characteristics] tone.",
            "This song’s segment in [Key] features a melody at a [pitch level] pitch and a [tempo] tempo for [duration], enveloped in [emotional characteristics] aura.",
            "The melody in this piece, set in [Key] with a [pitch level] pitch at a [tempo] tempo for [duration], exudes [emotional characteristics] aura.",
            "Flowing in [Key], the melody’s [pitch level] pitch and [tempo] tempo, lasting [duration], emanate [emotional characteristics] characteristics.",
            "The tune, positioned in [Key] with a [pitch level] pitch and a [tempo] tempo through [duration], conveys [emotional characteristics] tone.",
            "In the musical key of [Key], the melody’s [pitch level] pitch and [tempo] tempo over [duration] radiate [emotional characteristics] vibes.",
            "This melody, within the tonal scale of [Key], carrying a [pitch level] pitch and [tempo] tempo for [duration], emits [emotional characteristics] energy.",
            "Set to the harmonics of [Key], the melody, with a [pitch level] pitch and [tempo] tempo over [duration], embodies [emotional characteristics] aura.",
            "The melody from this segment, cast in [Key], holding a [pitch level] pitch and [tempo] tempo for [duration], reverberates with [emotional characteristics] mood.",
            "Within [Key], this melody’s [pitch level] pitch and [tempo] tempo, spanning [duration], are imbued with [emotional characteristics] atmosphere.",
            "The melody in this part of the song, set in [Key] with a [pitch level] pitch at a [tempo] tempo for [duration], envelops the listener with [emotional characteristics] atmosphere.",
            "The song, crafted in [Key], features a melody with a [pitch level] pitch and [tempo] tempo over [duration], filled with [emotional characteristics] emotions.",
            "In [Key], the melody strides with a [pitch level] pitch and [tempo] tempo for [duration], infused with [emotional characteristics] mood.",
            "This melody, anchored in [Key], progresses at a [pitch level] pitch and [tempo] tempo over [duration], draped in [emotional characteristics] aura.",
            "With its melody in [Key] at a [pitch level] pitch and moving at a [tempo] tempo for [duration], the song segment captures a [emotional characteristics] spirit.",
            "This segment's melody, from the key of [Key], maintains a [pitch level] pitch and [tempo] tempo throughout [duration], and exudes [emotional characteristics] aura.",
            "The tune, composed in [Key], with a pitch of [pitch level] and a tempo of [tempo] lasting [duration], radiates [emotional characteristics] emotions.",
            "Featuring a melody in [Key] with a [pitch level] pitch and [tempo] tempo over [duration], the song segment carries a [emotional characteristics] tone.",
            "Set in [Key], the melody, at a [pitch level] pitch and [tempo] tempo for [duration], projects [emotional characteristics] energy.",
            "The melody of this segment, playing in [Key] with a pitch at [pitch level] and tempo at [tempo] for [duration], is rich with [emotional characteristics] feels.",
            "This piece, tuned to [Key], plays a melody with a [pitch level] pitch and [tempo] tempo for [duration], delivering [emotional characteristics] emotions.",
            "In [Key], the melody rises with a [pitch level] pitch and flows at a [tempo] tempo for [duration], oozing [emotional characteristics] mood.",
            "This song segment, characterized by [Key] with a melody at a [pitch level] pitch and [tempo] tempo for [duration], emanates [emotional characteristics] characteristics.",
            "Set in the tonal realm of [Key], this melody’s [pitch level] pitch and [tempo] tempo through [duration] echo with [emotional characteristics] essence.",
            "The melody, flowing through [Key] with a [pitch level] pitch and [tempo] tempo for [duration], vibrates with [emotional characteristics] qualities.",
            "In [Key], this segment’s melody, at a [pitch level] pitch and a [tempo] tempo for [duration], pulses with [emotional characteristics] energy.",
            "The tune, nestled in [Key], at a [pitch level] pitch with a [tempo] tempo across [duration], embodies [emotional characteristics] aura.",
            "This musical segment in [Key], with a melody carrying a [pitch level] pitch and a [tempo] tempo for [duration], is steeped in [emotional characteristics] essence.",
            "Set within [Key], the melody at a [pitch level] pitch and [tempo] tempo, covering [duration], is suffused with [emotional characteristics] feel.",
            "This part of the song, in [Key] with a melody at a [pitch level] pitch and [tempo] tempo lasting [duration], carries a deep [emotional characteristics] feel.",
            "This melody, set in [Key] with a [pitch level] pitch and a [tempo] tempo, lasts for [duration] and is imbued with a [emotional characteristics] feel.",
            "In the key of [Key], this segment's melody has a [pitch level] pitch and a [tempo] tempo, spans [duration], and exudes [emotional characteristics] aura.",
            "Composed in [Key], the melody's pitch is [pitch level], moving at a [tempo] tempo for a duration of [duration], and carries a [emotional characteristics] mood.",
            "With its [pitch level] pitch and [tempo] tempo over [duration], the melody in [Key] radiates a [emotional characteristics] atmosphere.",
            "This piece in [Key] resonates with a [pitch level] pitch at a [tempo] tempo for [duration], filled with [emotional characteristics] emotions.",
            "The melody of this segment, crafted in [Key] with a [pitch level] pitch and a [tempo] tempo, lasting [duration], captures a [emotional characteristics] essence.",
            "Set in the tonal framework of [Key], this melody, having a [pitch level] pitch and a [tempo] pace, extends for [duration] and projects [emotional characteristics] energy.",
            "The melody, originating in [Key] at a [pitch level] pitch and moving at a [tempo] tempo for [duration], pulses with [emotional characteristics] emotions.",
            "This segment’s melody in [Key], with its [pitch level] pitch and [tempo] tempo across [duration], breathes a [emotional characteristics] essence.",
            "In [Key], this melody with a [pitch level] pitch and [tempo] tempo over [duration], is steeped in [emotional characteristics] essence.",
            "The tune, rooted in [Key] and exhibiting a [pitch level] pitch with a [tempo] tempo for [duration], vibrates with [emotional characteristics] qualities.",
            "This melody, from the key of [Key], carrying a [pitch level] pitch and progressing at a [tempo] pace for [duration], is laden with [emotional characteristics] essence.",
            "With a [pitch level] pitch and [tempo] tempo, this melody from [Key], lasting [duration], channels [emotional characteristics] emotions.",
            "The song, played in [Key], maintains a [pitch level] pitch and [tempo] tempo through its [duration], resonating with [emotional characteristics] mood.",
            "This piece’s melody, in [Key] at a [pitch level] pitch and a [tempo] tempo over [duration], manifests a [emotional characteristics] ambiance.",
            "The melody, echoing through [Key] with a [pitch level] pitch and [tempo] tempo for [duration], is filled with [emotional characteristics] emotions.",
            "Set in [Key], the melody’s [pitch level] pitch and [tempo] tempo, spanning [duration], evoke [emotional characteristics] feels.",
            "In [Key], this song's segment offers a melody with a [pitch level] pitch at a [tempo] pace for [duration], imbued with [emotional characteristics] feel.",
            "This segment, orchestrated in [Key], utilizes a melody with a [pitch level] pitch and a [tempo] tempo lasting [duration], enriched by [emotional characteristics] emotions.",
            "Composed in [Key], the melody upholds a [pitch level] pitch and a [tempo] pace through a [duration], suffused with [emotional characteristics] feel.",
            "The melody, unfolding in [Key] with a [pitch level] pitch and a [tempo] tempo for [duration], showcases [emotional characteristics] tone.",
            "This song’s segment in [Key] features a melody at a [pitch level] pitch and a [tempo] tempo for [duration], enveloped in [emotional characteristics] aura.",
            "The melody in this piece, set in [Key] with a [pitch level] pitch at a [tempo] tempo for [duration], exudes [emotional characteristics] aura.",
            "Flowing in [Key], the melody’s [pitch level] pitch and [tempo] tempo, lasting [duration], emanate [emotional characteristics] characteristics.",
            "The tune, positioned in [Key] with a [pitch level] pitch and a [tempo] tempo through [duration], conveys [emotional characteristics] tone.",
            "In the musical key of [Key], the melody’s [pitch level] pitch and [tempo] tempo over [duration] radiate [emotional characteristics] atmosphere.",
            "This melody, within the tonal scale of [Key], carrying a [pitch level] pitch and [tempo] tempo for [duration], emits [emotional characteristics] energy.",
            "Set to the harmonics of [Key], the melody, with a [pitch level] pitch and [tempo] tempo over [duration], embodies [emotional characteristics] aura.",
            "The melody from this segment, cast in [Key], holding a [pitch level] pitch and [tempo] tempo for [duration], reverberates with [emotional characteristics] mood.",
            "Within [Key], this melody’s [pitch level] pitch and [tempo] tempo, spanning [duration], are imbued with [emotional characteristics] feel.",
            "The melody in this part of the song, set in [Key] with a [pitch level] pitch at a [tempo] tempo for [duration], envelops the listener with [emotional characteristics] atmosphere.",
            "The song, crafted in [Key], features a melody with a [pitch level] pitch and [tempo] tempo over [duration], filled with [emotional characteristics] emotions emotions.",
            "In [Key], the melody strides with a [pitch level] pitch and flows at a [tempo] tempo for [duration], infused with [emotional characteristics] mood.",
            "This melody, anchored in [Key], progresses at a [pitch level] pitch and [tempo] tempo over [duration], draped in [emotional characteristics] aura.",
            "With its melody in [Key] at a [pitch level] pitch and moving at a [tempo] tempo for [duration], the song segment captures a [emotional characteristics] spirit.",
            "This segment's melody, from the key of [Key], maintains a [pitch level] pitch and [tempo] tempo throughout [duration], and exudes [emotional characteristics] aura.",
            "The tune, composed in [Key], with a pitch of [pitch level] and a tempo of [tempo] lasting [duration], radiates [emotional characteristics] emotions.",
            "Featuring a melody in [Key] with a [pitch level] pitch and [tempo] tempo over [duration], the song segment carries a [emotional characteristics] tone.",
            "Set in [Key], the melody, at a [pitch level] pitch and [tempo] tempo for [duration], projects [emotional characteristics] energy.",
            "The melody of this segment, playing in [Key] with a pitch at [pitch level] and tempo at [tempo] for [duration], is rich with [emotional characteristics] emotions.",
            "This piece, tuned to [Key], plays a melody with a [pitch level] pitch and [tempo] tempo for [duration], delivering [emotional characteristics] qualities.",
            "In [Key], the melody rises with a [pitch level] pitch and flows at a [tempo] tempo for [duration], oozing [emotional characteristics] mood.",
            "This song segment, characterized by [Key] with a melody at a [pitch level] pitch and [tempo] tempo for [duration], emanates [emotional characteristics] characteristics.",
            "Set in the tonal realm of [Key], this melody’s [pitch level] pitch and [tempo] tempo through [duration] echo with [emotional characteristics] essence.",
            "The melody, flowing through [Key] with a [pitch level] pitch and [tempo] tempo for [duration], vibrates with [emotional characteristics] qualities qualities.",
            "In [Key], this segment’s melody, at a [pitch level] pitch and a [tempo] tempo for [duration], pulses with [emotional characteristics] emotions.",
            "The tune, nestled in [Key], at a [pitch level] pitch with a [tempo] tempo across [duration], embodies [emotional characteristics] aura.",
            "This musical segment in [Key], with a melody carrying a [pitch level] pitch and a [tempo] tempo for [duration], is steeped in [emotional characteristics] essence.",
            "Set within [Key], the melody at a [pitch level] pitch and [tempo] tempo, covering [duration], is suffused with [emotional characteristics] feel.",
            "This part of the song, in [Key] with a melody at a [pitch level] pitch and [tempo] tempo lasting [duration], carries a deep [emotional characteristics] feel.",
        ]

        self.templates_wo_key_w_duration = [
            "This melody, characterized by a [pitch level] pitch and a [tempo] tempo, lasts [duration] and envelops the listener with a [emotional characteristics] feel.",
            "With a pitch of [pitch level] and a tempo of [tempo], this song segment spans a brief [duration], radiating a [emotional characteristics] mood.",
            "The melody of this segment, having a [pitch level] pitch and moving at a [tempo] pace, extends over [duration] and is imbued with [emotional characteristics] feel.",
            "This piece features a melody at a [pitch level] pitch with a [tempo] tempo, covering [duration] period, and filled with [emotional characteristics] emotions.",
            "Set with a [pitch level] pitch and a tempo of [tempo], this melody’s duration of [duration] carries a distinctly [emotional characteristics] atmosphere.",
            "The song, flowing at a [tempo] tempo with a [pitch level] pitch for [duration], exudes a [emotional characteristics] vibe throughout.",
            "In this music segment, the melody soars to a [pitch level] pitch, moves at a [tempo] tempo, lasts [duration], and creates a [emotional characteristics] ambiance.",
            "This melody, at a [pitch level] pitch and pacing at [tempo], spans a duration of [duration] and is charged with [emotional characteristics] energy.",
            "The tune, lasting [duration] with a [pitch level] pitch and a moderate [tempo] speed, evokes a [emotional characteristics] sentiment.",
            "With [duration], the melody in this song part, pitched at [pitch level] and set to a [tempo] tempo, pulses with [emotional characteristics] tones.",
        ]

        self.templates_wo_avg_pitch_w_duration = [
            "This melody, set in [Key], moves at a [tempo] pace and lasts for [duration], enveloped in a [emotional characteristics] atmosphere.",
            "In the key of [Key], the segment features a melody with a medium tempo and [duration], radiating a [emotional characteristics] mood.",
            "The melody of this song, crafted in [Key], progresses at a [tempo] speed and spans [duration], filled with [emotional characteristics] emotions.",
            "This song's melody, tuned to [Key] and pacing at a [tempo] rate, continues for [duration] and conveys a distinctly [emotional characteristics] feel.",
            "Set in [Key], this short melody lasts [duration] and moves with a [tempo] tempo, capturing a [emotional characteristics] essence.",
            "The tune in [Key] unfolds at a [tempo] tempo for [duration], evoking a [emotional characteristics] sentiment throughout.",
            "Flowing through [Key] at a [tempo] tempo, this melody lasts [duration] and carries a rich [emotional characteristics] ambiance.",
            "With [duration], the melody in this segment of the song in [Key] at a [tempo] pace exudes a [emotional characteristics] atmosphere.",
            "This melody, set in the key of [Key], extends for [duration] and operates at a [tempo] tempo, steeped in a [emotional characteristics] vibe.",
            "In the musical context of [Key], this brief melody, lasting [duration] and moving at a [tempo] tempo, is imbued with a [emotional characteristics] feel.",
        ]

        self.templates_wo_tempo_w_duration = [
            "This song segment, composed in [Key], features a melody with a [pitch level] pitch, spans [duration], and exudes a [emotional characteristics] atmosphere.",
            "In the key of [Key], this melody rises to a [pitch level] pitch, lasts for [duration], and carries a [emotional characteristics] mood.",
            "Set in [Key], the melody’s [pitch level] pitch and brief duration of [duration] bring out a [emotional characteristics] ambiance.",
            "The tune, crafted in [Key], achieves a [pitch level] pitch over [duration], filled with [emotional characteristics] emotions.",
            "With its high pitch and [duration] duration, this melody in [Key] radiates a distinctly [emotional characteristics] feel.",
            "This melody, played in [Key] with a [pitch level] pitch for [duration], manifests a rich [emotional characteristics] vibe.",
            "The [Key] key hosts this melody, reaching a [pitch level] pitch, spanning [duration], and echoing [emotional characteristics] feels.",
            "This short melody, set in [Key] and reaching a [pitch level] pitch, captures [emotional characteristics] essence over [duration].",
            "Within [Key], the melody, at a [pitch level] pitch for [duration], pulses with [emotional characteristics] tones.",
            "Flowing through [Key] at a [pitch level] pitch, this melody lasts [duration] and exudes [emotional characteristics] feels.",
        ]

        self.templates_wo_emotion_w_duration = [
            "This song segment, set in [Key], features a [pitch level] pitch, [tempo] tempo, and lasts [duration].",
            "Composed in [Key], the melody has a [pitch level] pitch, moves at a [tempo] pace, and spans [duration].",
            "The melody in [Key] reaches a [pitch level] pitch, flows at a [tempo] tempo, and finishes in [duration].",
            "Set in the key of [Key], this segment presents a [pitch level] pitch, a [tempo] tempo, and lasts [duration].",
            "With a [pitch level] pitch and [tempo] tempo, this melody in [Key] lasts for [duration].",
            "In [Key], the tune ascends to a [pitch level] pitch, travels at a [tempo] tempo, and ends within [duration].",
            "This piece, played in [Key], holds a [pitch level] pitch and a [tempo] tempo over [duration].",
            "The segment’s melody, in the key of [Key], with a [pitch level] pitch and [tempo] tempo, extends for [duration].",
            "Composed in [Key] with a [pitch level] pitch and medium [tempo], this melody concludes in [duration].",
            "Featuring a [pitch level] pitch and a [tempo] tempo, the melody from [Key] covers a span of [duration].",
            "In [Key], this song segment maintains a [pitch level] pitch and a [tempo] tempo for a duration of [duration].",
            "The melody, orchestrated in [Key], strikes a [pitch level] pitch, matches a [tempo] tempo, and lasts [duration].",
            "With its [pitch level] pitch and [tempo] tempo, this [Key] melody concludes quickly, in [duration].",
            "This melody from [Key] features a [pitch level] pitch, a [tempo] tempo, and a quick finish after [duration].",
            "The tune, set in [Key], reaches a [pitch level] pitch and moderates at a [tempo] tempo for [duration].",
            "In the key of [Key], the melody’s [pitch level] pitch and [tempo] tempo span [duration].",
            "This short melody in [Key] with a [pitch level] pitch and a [tempo] tempo wraps up in [duration].",
            "The song’s section in [Key] maintains a [pitch level] pitch, a [tempo] pace, and [duration].",
            "Set within [Key], the melody at a [pitch level] pitch moves through its [tempo] tempo in [duration].",
            "A melody in [Key] with a [pitch level] pitch and [tempo] tempo unfolds and completes in [duration].",
        ]

        self.templates_wo_key_and_avg_pitch_w_duration = [
            "This melody, with a [tempo] tempo and lasting [duration], exudes a [emotional characteristics] feel throughout.",
            "Set to a [tempo] pace and spanning [duration], this song segment carries a distinctly [emotional characteristics] ambiance.",
            "The melody flows at a [tempo] tempo, finishes quickly within [duration], and is infused with a [emotional characteristics] mood.",
            "With its duration of [duration] and a [tempo] tempo, the melody creates a [emotional characteristics] atmosphere.",
            "This song piece, moving at a [tempo] tempo and concluding in [duration], radiates [emotional characteristics] vibes.",
            "In [duration], the melody at a [tempo] tempo envelops the listener with its [emotional characteristics] essence.",
            "The melody's [tempo] tempo over a span of [duration] brings out a [emotional characteristics] emotional tone.",
            "This brief melody, lasting [duration] at a [tempo] pace, is filled with [emotional characteristics] emotions.",
            "Set at a [tempo] tempo and wrapping up in [duration], this segment is imbued with a [emotional characteristics] feel.",
            "The tune, flowing at a [tempo] pace for [duration], captures a deeply [emotional characteristics] sentiment.",
        ]

        self.templates_wo_key_and_tempo_w_duration = [
            "This melody, with a [pitch level] pitch and lasting [duration], is imbued with a [emotional characteristics] ambiance.",
            "Featuring a [pitch level] pitch and a duration of [duration], the melody exudes a [emotional characteristics] mood.",
            "The segment, characterized by a [pitch level] pitch and [duration], radiates a [emotional characteristics] feel.",
            "With its [pitch level] pitch and duration of [duration], this melody carries a [emotional characteristics] atmosphere.",
            "This melody's [pitch level] pitch over a span of [duration] creates a distinctly [emotional characteristics] emotional tone.",
            "The song, pitched at [pitch level] for a duration of [duration], is filled with a [emotional characteristics] essence.",
            "In [duration], this high-pitched melody cultivates a [emotional characteristics] environment.",
            "The tune, holding a [pitch level] pitch for [duration], is steeped in a [emotional characteristics] vibe.",
            "This short melody, resonating at a [pitch level] pitch for [duration], encapsulates a [emotional characteristics] sentiment.",
            "Set at a [pitch level] pitch and concluding in [duration], this segment beautifully conveys a [emotional characteristics] mood.",
        ]

        self.templates_wo_key_and_emotion_w_duration = [
            "This melody, with a [pitch level] pitch and a [tempo] tempo, spans [duration].",
            "Set to a [pitch level] pitch and moving at a [tempo] pace, this segment lasts for [duration].",
            "The song's segment, characterized by a [pitch level] pitch and a moderate [tempo] speed, finishes within [duration].",
            "With its pitch at [pitch level] and a tempo of [tempo], this melody concludes in [duration].",
            "This brief musical piece carries a [pitch level] pitch and maintains a [tempo] tempo throughout [duration].",
            "Featuring a [pitch level] pitch and a [tempo] tempo, this melody's entire span is [duration].",
            "The tune, high-pitched at [pitch level] and set to a [tempo] tempo, wraps up in [duration].",
            "In [duration], this melody with a [pitch level] pitch and [tempo] speed expresses its brief but memorable theme.",
            "This segment's melody, pitched at [pitch level] and with a [tempo] pacing, lasts [duration].",
            "The melody's high [pitch level] pitch coupled with its [tempo] tempo, spans [duration].",
        ]

        self.templates_wo_avg_pitch_and_tempo_w_duration = [
            "Set in [Key], this melody lasts [duration] and is imbued with a [emotional characteristics] feel.",
            "The song segment, in the key of [Key], spans [duration] and carries a distinctly [emotional characteristics] atmosphere.",
            "This short melody, composed in [Key], resonates with [emotional characteristics] tones for [duration].",
            "Lasting [duration], the melody in [Key] evokes a deeply [emotional characteristics] mood.",
            "With [duration], the melody in [Key] exudes a [emotional characteristics] ambiance throughout.",
            "This piece, set in [Key] and lasting [duration], radiates with [emotional characteristics] emotions.",
            "The melody, crafted in [Key] and spanning [duration], conveys a rich [emotional characteristics] feel.",
            "In [Key], this brief melody extends for [duration] and is steeped in [emotional characteristics] emotions.",
            "This melody, resonating in [Key] for [duration], is filled with a [emotional characteristics] essence.",
            "Featuring a melody in [Key] that lasts [duration], this segment captures a [emotional characteristics] sentiment effectively.",
        ]

        self.templates_wo_avg_pitch_and_emotion_w_duration = [
            "This melody, set in [Key], flows at a [tempo] tempo and lasts [duration].",
            "In the key of [Key], the segment progresses at a [tempo] pace, spanning [duration].",
            "The tune in [Key] maintains a [tempo] tempo for [duration].",
            "With a [tempo] tempo in [Key], this melody concludes in [duration].",
            "This song, played in [Key], moves at a [tempo] pace and wraps up in [duration].",
            "The melody, set to the key of [Key], lasts [duration] and has a [tempo] speed.",
            "This short melody in [Key] at a [tempo] tempo runs for [duration].",
            "In [Key], this segment featuring a [tempo] tempo finishes after [duration].",
            "The melody, resonating in [Key], with a [tempo] tempo, concludes in [duration].",
            "Composed in [Key], this piece keeps a [tempo] tempo over [duration].",
        ]

        self.templates_wo_tempo_and_emotion_w_duration = [
            "This segment, set in [Key] with a [pitch level] pitch, plays out over [duration].",
            "The melody in [Key], featuring a [pitch level] pitch, lasts for [duration].",
            "In the key of [Key], this tune rises to a [pitch level] pitch and wraps up within [duration].",
            "Set within [Key], the melody, which reaches a [pitch level] pitch, concludes after [duration].",
            "This musical piece in [Key] carries a [pitch level] pitch and spans [duration].",
            "In [Key], the segment's high [pitch level] pitch carries through [duration].",
            "The tune, pitched at [pitch level] in [Key], plays for [duration].",
            "Featuring a [pitch level] pitch in [Key], this melody concludes swiftly, lasting [duration].",
            "This short melody in [Key], having a [pitch level] pitch, extends for [duration].",
            "The piece, set in [Key] with a pitch of [pitch level], is brief, enduring [duration].",
        ]

        self.templates_wo_key_and_avg_pitch_and_tempo_w_duration = [
            "This melody unfolds over a duration of [duration], carrying a [emotional characteristics] vibe that complements its medium pace."
            "Lasting [duration], the tune progresses at a moderate speed, richly infused with a [emotional characteristics] atmosphere."
            "The melody, extending for [duration] and moving at a medium tempo, is steeped in [emotional characteristics] emotions."
            "Over the course of [duration], this melody, paced moderately, exudes a deeply [emotional characteristics] mood."
            "With [duration] and a medium tempo, this segment resonates with a distinctly [emotional characteristics] feel."
        ]

        self.templates_wo_key_and_avg_pitch_and_emotion_w_duration = [
            "This melody, with a [tempo] tempo, concludes within a span of [duration].",
            "Lasting [duration], the tune progresses at a [tempo] speed, capturing the essence of brevity.",
            "The melody, moving at a [tempo] pace, lasting [duration].",
            "Set to a [tempo] tempo, this segment of the song spans [duration].",
            "With [duration] and a [tempo] tempo, this melody succinctly delivers its musical message.",
        ]

        # ################ without duration information ################

        self.full_factor_templates = [
            "This melody, set in [Key], features a [pitch level] pitch and [tempo] tempo, rich in [emotional characteristics] emotions.",
            "In [Key], the segment's melody has a [pitch level] pitch and [tempo] tempo, exuding [emotional characteristics] feel.",
            "With a melody in [Key], this segment conveys a [pitch level] pitch and [tempo] tempo, filled with [emotional characteristics] tones.",
            "The song progresses in [Key] with a melody that hits a [pitch level] pitch and a [tempo] tempo, highlighting [emotional characteristics] aura.",
            "Playing in [Key], the melody travels through a [pitch level] pitch and [tempo] tempo, emphasizing [emotional characteristics] qualities.",
            "This musical piece in [Key] maintains a melody with a [pitch level] pitch and [tempo] tempo, encapsulating [emotional characteristics] characteristics.",
            "Set against a backdrop of [Key], the melody flows with a [pitch level] pitch and a [tempo] tempo, suffused with [emotional characteristics] feel.",
            "The tune, rooted in [Key], holds a [pitch level] pitch and [tempo] tempo, infused with [emotional characteristics] emotions.",
            "In the key of [Key], this segment presents a melody with a [pitch level] pitch and [tempo] tempo, pulsating with [emotional characteristics] mood.",
            "This part of the song, in [Key], showcases a melody that carries a [pitch level] pitch and [tempo] tempo, rich with [emotional characteristics] mood.",
            "Orchestrated in [Key], the melody achieves a [pitch level] pitch and operates at a [tempo] tempo, resonating with [emotional characteristics] qualities.",
            "The melody in this song section, composed in [Key], displays a [pitch level] pitch and [tempo] tempo, draped in [emotional characteristics] aura.",
            "With its roots in [Key], the segment plays out at a [pitch level] pitch and [tempo] tempo, teeming with [emotional characteristics] mood.",
            "This melody, orchestrated in [Key], balances a [pitch level] pitch with a [tempo] tempo, enveloped in [emotional characteristics] aura.",
            "Featured in the key of [Key], this melody has a pitch of [pitch level] and a tempo of [tempo], bristling with [emotional characteristics] feel.",
            "The song’s melody, crafted in [Key], holds a pitch of [pitch level] and moves at a tempo of [tempo], brewing with [emotional characteristics] emotions.",
            "Composed in [Key], the melody spans a [pitch level] pitch and a [tempo] tempo, bursting with [emotional characteristics] energy.",
            "The melodic line in [Key] carries a [pitch level] pitch and moves at a [tempo] tempo, laden with [emotional characteristics] essence.",
            "Situated in [Key], this melody carries a [pitch level] pitch and maintains a [tempo] tempo, imbued with [emotional characteristics] emotions.",
            "The tune, set in [Key], progresses with a [pitch level] pitch and a [tempo] tempo, flourishing with [emotional characteristics] characteristics.",
            "This segment's melody, composed in [Key], has a pitch of [pitch level] and tempo of [tempo], adorned with [emotional characteristics] mood.",
            "Melody in [Key] with a pitch of [pitch level] and a tempo of [tempo], radiates [emotional characteristics] vibes.",
            "With its melody in [Key], this song segment uses a [pitch level] pitch and [tempo] tempo, pulsing with [emotional characteristics] energy.",
            "Set in the tonal center of [Key], the melody spans a [pitch level] pitch and moves with a [tempo] tempo, echoing [emotional characteristics] essence.",
            "The melody of this part of the song, set in [Key], features a [pitch level] pitch and a [tempo] tempo, adorned with [emotional characteristics] feel.",
            "This song section, with its melody in [Key], holds a [pitch level] pitch and a [tempo] tempo, charged with [emotional characteristics] energy.",
            "The segment, carried by a melody in [Key], features a [pitch level] pitch and a [tempo] tempo, saturated with [emotional characteristics] essence.",
            "This piece of the melody, keyed in [Key], supports a [pitch level] pitch and a [tempo] tempo, enveloped by [emotional characteristics] aura.",
            "Composed in [Key], this melody's pitch level of [pitch level] and tempo of [tempo] are infused with [emotional characteristics] emotions.",
            "The melody, situated in [Key], proceeds with a [pitch level] pitch and a [tempo] tempo, manifesting [emotional characteristics] mood.",
            "In the musical key of [Key], the melody moves with a [pitch level] pitch and a [tempo] tempo, bursting with [emotional characteristics] energy.",
            "This song's melody, in the key of [Key], maintains a [pitch level] pitch and a [tempo] tempo, dripping with [emotional characteristics] essence.",
            "Rooted in [Key], the melody of this segment follows a [pitch level] pitch and [tempo] tempo, filled with [emotional characteristics] energy.",
            "The melodic contour in [Key] unfolds at a [pitch level] pitch and a [tempo] tempo, filled with [emotional characteristics] emotions emotions.",
            "This song's segment, keyed in [Key], parades a melody at a [pitch level] pitch and a [tempo] tempo, heavy with [emotional characteristics] qualities.",
            "Within the key of [Key], the segment's melody maintains a [pitch level] pitch and [tempo] tempo, alive with [emotional characteristics] energy.",
            "Orchestrated in [Key], this melody sweeps through a [pitch level] pitch and a [tempo] tempo, echoing with [emotional characteristics] mood.",
            "In [Key], the melody of this segment carries a [pitch level] pitch and strides at a [tempo] tempo, vibrant with [emotional characteristics].",
            "This melody, resonating in [Key], claims a [pitch level] pitch and a [tempo] tempo, permeated with [emotional characteristics] energy.",
            "The segment, set in [Key], delivers a melody at a [pitch level] pitch and a [tempo] tempo, resonant with [emotional characteristics] character.",
            "In the musical landscape of [Key], this melody carries a [pitch level] pitch and [tempo] tempo, thrumming with [emotional characteristics] emotions.",
            "This musical excerpt, in [Key], offers a melody with a [pitch level] pitch and a [tempo] tempo, cloaked in [emotional characteristics] aura.",
            "The melody of this song section, crafted in [Key], operates at a [pitch level] pitch and [tempo] tempo, echoing with [emotional characteristics] feel.",
            "Featured in the tonal framework of [Key], the melody moves at a [pitch level] pitch and [tempo] tempo, animated by [emotional characteristics] nature.",
            "In [Key], this song segment’s melody unfolds with a [pitch level] pitch and [tempo] tempo, laden with [emotional characteristics] feel.",
            "The melody, structured in [Key], ascends to a [pitch level] pitch and proceeds at a [tempo] tempo, drenched in [emotional characteristics] emotions.",
            "This segment’s melody, in [Key], progresses with a [pitch level] pitch and a [tempo] tempo, imbued with [emotional characteristics] atmosphere.",
            "The melody of this part, keyed in [Key], holds a [pitch level] pitch and [tempo] tempo, ripe with [emotional characteristics] essence.",
            "Composed in [Key], the melody reaches a [pitch level] pitch and moves at a [tempo] tempo, rife with [emotional characteristics] characteristics.",
            "This part of the song, melodically set in [Key], resonates at a [pitch level] pitch and [tempo] tempo, heavy with [emotional characteristics] mood.",
            "This melody, composed in [Key] with a [pitch level] pitch and [tempo] tempo, brims with [emotional characteristics] emotions.",
            "The segment, set in [Key], features a [pitch level] pitch and [tempo] pace, echoing [emotional characteristics] feel.",
            "In [Key], this melody's pitch is [pitch level] and its tempo is [tempo], filled with [emotional characteristics] qualities.",
            "With a pitch of [pitch level] and a tempo of [tempo], the tune in [Key] exudes [emotional characteristics] aura.",
            "This song part, tuned to [Key] and moving at a [tempo] tempo, carries a [pitch level] pitch and [emotional characteristics] tone.",
            "The melody's [pitch level] pitch and [tempo] tempo, set in [Key], perfectly capture its [emotional characteristics] nature.",
            "This tune, keyed in [Key], unfolds with a [pitch level] pitch and a rhythm of [tempo], radiating [emotional characteristics] vibes.",
            "Set in the musical scale of [Key], the melody strikes a [pitch level] pitch and moves with a [tempo] tempo, bursting with [emotional characteristics] energy.",
            "The piece, composed in [Key] with a pitch of [pitch level] and a tempo described as [tempo], showcases [emotional characteristics] vibe.",
            "With its rich [emotional characteristics] nature, the melody in [Key] carries a [pitch level] pitch and a tempo of [tempo].",
            "In [Key], the song's melody operates at a [pitch level] pitch and a [tempo] pace, infused with [emotional characteristics] tone.",
            "This song section in [Key] holds a [pitch level] pitch and a tempo termed [tempo], wrapped in [emotional characteristics] emotions.",
            "The melody, crafted in [Key] with a pitch of [pitch level] and a tempo of [tempo], vibrates with [emotional characteristics] mood.",
            "The tune, emanating from [Key] with a [pitch level] pitch and [tempo] tempo, pulses with [emotional characteristics] energy.",
            "This musical segment in [Key] with a [pitch level] pitch and a tempo described as [tempo], teems with [emotional characteristics] characteristics.",
            "Rendered in [Key], the melody achieves a pitch of [pitch level] and a [tempo] tempo, exuding [emotional characteristics] emotions.",
            "In [Key], the song’s melody reaches a [pitch level] pitch and progresses at a [tempo] pace, filled with [emotional characteristics] quality.",
            "The melody, anchored in [Key], maintains a [pitch level] pitch and operates at a [tempo] speed, charged with [emotional characteristics] energy.",
            "With a pitch of [pitch level] in [Key] and a tempo of [tempo], this melody is laden with [emotional characteristics] emotions.",
            "The tune, crafted in [Key] at a pitch of [pitch level] and a [tempo] tempo, is imbued with [emotional characteristics] tone.",
            "In the key of [Key], this melody’s pitch of [pitch level] and tempo of [tempo] convey [emotional characteristics] tone.",
            "This segment’s melody in [Key], with its [pitch level] pitch and [tempo] tempo, is bursting with [emotional characteristics] tone.",
            "Set to the tune of [Key] with a [pitch level] pitch and a [tempo] tempo, the melody thrives with [emotional characteristics] feel.",
            "The musical piece in [Key], featuring a pitch of [pitch level] and a tempo of [tempo], showcases [emotional characteristics] vibe.",
            "This melody, resonating in [Key] at a [pitch level] pitch and [tempo] tempo, is rich with [emotional characteristics] characteristics.",
            "Rendered in [Key], the tune’s pitch of [pitch level] and tempo of [tempo] resonate with [emotional characteristics] mood.",
            "Composed in [Key] with a pitch of [pitch level] and a [tempo] tempo, the melody is vibrant with [emotional characteristics] emotions.",
            "In [Key], this melody navigates a pitch of [pitch level] and a tempo of [tempo], steeped in [emotional characteristics] essence.",
            "This segment, keyed in [Key], moves at a [tempo] tempo with a pitch of [pitch level], colored by [emotional characteristics] qualities.",
            "The tune, operating in [Key] at a [pitch level] pitch and [tempo] tempo, is saturated with [emotional characteristics] feel.",
            "Set in the harmonic scale of [Key], this melody achieves a pitch of [pitch level] and a tempo of [tempo], teeming with [emotional characteristics] emotions.",
            "This melody, structured in [Key], combines a pitch of [pitch level] with a tempo of [tempo], emanating [emotional characteristics] essence.",
            "With its pitch set at [pitch level] and its tempo at [tempo], this melody in [Key] embodies [emotional characteristics] nature.",
            "In the key of [Key], this song’s segment pulses at a [pitch level] pitch and [tempo] tempo, oozing [emotional characteristics] mood.",
            "This melody, in [Key] and having a pitch of [pitch level] and a tempo of [tempo], vibrates with [emotional characteristics] qualities.",
            "This tune in [Key], characterized by a [pitch level] pitch and [tempo] tempo, radiates [emotional characteristics] emotions.",
            "The song, set in [Key] with a pitch of [pitch level] and a tempo of [tempo], thrives with [emotional characteristics] feel.",
            "Orchestrated in [Key], this melody holds a pitch of [pitch level] and a tempo of [tempo], brimming with [emotional characteristics] essence.",
            "In the tonality of [Key], the melody unfolds with a pitch of [pitch level] and a tempo of [tempo], full of [emotional characteristics] emotions.",
            "This song section, with its melody in [Key] at a pitch of [pitch level] and tempo of [tempo], pulses with [emotional characteristics] energy.",
            "Set in [Key], this tune progresses with a pitch of [pitch level] and a tempo of [tempo], echoing [emotional characteristics] tone.",
            "Composed in [Key], the melody’s pitch of [pitch level] and tempo of [tempo] are infused with [emotional characteristics] emotions.",
            "This segment, rendered in [Key], presents a melody at a pitch of [pitch level] and a tempo of [tempo], steeped in [emotional characteristics] essence.",
            "The melody, flowing in [Key] at a pitch of [pitch level] and a tempo of [tempo], showcases [emotional characteristics] tone.",
            "In the musical landscape of [Key], this melody carries a pitch of [pitch level] and a tempo of [tempo], rich with [emotional characteristics] qualities.",
            "This melody, tuned in [Key] at a pitch of [pitch level] and a [tempo] pace, conveys [emotional characteristics] emotions.",
            "Set within the tonal framework of [Key], the melody reaches a pitch of [pitch level] and progresses at a tempo of [tempo], filled with [emotional characteristics] essence.",
            "The song’s melody, crafted in [Key], maintains a pitch of [pitch level] and a tempo of [tempo], resonating with [emotional characteristics] mood.",
            "This segment of the song, keyed in [Key] with a [pitch level] pitch and [tempo] tempo, thrums with [emotional characteristics] essence.",
            "Composed in [Key] and moving at a tempo of [tempo], this melody, at a pitch of [pitch level], is vibrant with [emotional characteristics] tone.",
        ]

        self.templates_wo_key = [
            "This melody carries a [pitch level] pitch and moves at a [tempo] tempo, bursting with [emotional characteristics] energy.",
            "The tune operates with a [pitch level] pitch and a [tempo] tempo, richly imbued with [emotional characteristics] spirit.",
            "With its [pitch level] pitch and [tempo] tempo, this melody exudes a [emotional characteristics] aura.",
            "The melody holds a [pitch level] pitch and maintains a [tempo] tempo, conveying deep [emotional characteristics] mood.",
            "This musical piece, characterized by a [pitch level] pitch and a [tempo] tempo, radiates with [emotional characteristics] emotions.",
            "The melody here unfolds at a [pitch level] pitch and progresses at a [tempo] tempo, filled with [emotional characteristics] characteristics.",
            "Featuring a [pitch level] pitch and a [tempo] tempo, this segment of the song resonates with [emotional characteristics] essence.",
            "This song's melody, with a [pitch level] pitch and [tempo] tempo, vibrates with a sense of [emotional characteristics] mood.",
            "With a pitch of [pitch level] and a tempo of [tempo], this melody channels [emotional characteristics] emotions.",
            "The tune, carrying a pitch of [pitch level] and pacing at [tempo], thrives with [emotional characteristics] feel.",
        ]

        self.templates_wo_avg_pitch = [
            "This melody, set in [Key], moves at a [tempo] pace and exudes a [emotional characteristics] feel.",
            "In the key of [Key], the melody unfolds with a [tempo] tempo, filled with [emotional characteristics] energy.",
            "The tune, played in [Key], maintains a [tempo] tempo and is charged with [emotional characteristics] energy.",
            "With a tempo of [tempo], the melody in [Key] vibrates with an air of [emotional characteristics] characteristics.",
            "The melody, characterized by its [Key] tonality and [tempo] speed, carries [emotional characteristics] tone.",
            "This segment, composed in [Key] and moving at a [tempo] pace, pulses with [emotional characteristics] emotions.",
            "Flowing in the key of [Key] at a [tempo] tempo, this melody radiates [emotional characteristics] vibes.",
            "Set in [Key] with a tempo of [tempo], the melody here is steeped in [emotional characteristics] feel.",
            "This musical piece, performed in [Key] at a [tempo] pace, conveys [emotional characteristics] tone.",
            "The tune, originating in [Key] and progressing at a [tempo] rate, is imbued with [emotional characteristics] spirit.",
        ]

        self.templates_wo_tempo = [
            "Set in [Key] with a [pitch level] pitch, the melody thrives with [emotional characteristics] energy.",
            "In the key of [Key], this melody, which carries a [pitch level] pitch, is rich in [emotional characteristics] mood.",
            "This segment's melody, tuned to [Key] and possessing a [pitch level] pitch, exudes a [emotional characteristics] aura.",
            "With its pitch at [pitch level] and in the key of [Key], the melody pulsates with [emotional characteristics] energy.",
            "The melody, set in [Key] with a [pitch level] pitch, vividly captures the [emotional characteristics] nature.",
            "Composed in [Key] and marked by a [pitch level] pitch, this melody resonates with [emotional characteristics] emotions.",
            "This tune, resonating in [Key] at a [pitch level] pitch, radiates [emotional characteristics] vibes.",
            "Carrying a [pitch level] pitch in the key of [Key], the melody is imbued with [emotional characteristics] atmosphere.",
            "Set in the tonal realm of [Key] and featuring a [pitch level] pitch, the melody channels [emotional characteristics] essence.",
            "The melody, with its [pitch level] pitch in [Key], vibrates intensely with [emotional characteristics] feel.",
        ]

        self.templates_wo_emotion = [
            "This melody, in the key of [Key], features a [pitch level] pitch and progresses at a [tempo] tempo.",
            "Set in [Key], the melody carries a [pitch level] pitch and maintains a [tempo] pace.",
            "Composed in [Key], this segment’s pitch is [pitch level] with a tempo that is [tempo].",
            "In [Key], the melody achieves a [pitch level] pitch and moves at a [tempo] speed.",
            "The tune, characterized by a [pitch level] pitch and a [tempo] tempo, is set in the key of [Key].",
            "This song’s segment, tuned to [Key], presents a [pitch level] pitch and a tempo of [tempo].",
            "With a pitch of [pitch level] and a tempo of [tempo], this melody in [Key] flows smoothly.",
            "In the key of [Key], the melody resonates at a [pitch level] pitch and operates at a [tempo] tempo.",
            "This piece, set in [Key], unfolds with a pitch of [pitch level] and a [tempo] rhythm.",
            "The melody, anchored in [Key], spans a pitch of [pitch level] and a pace of [tempo].",
            "Cast in the musical key of [Key], this melody holds a [pitch level] pitch and a [tempo] tempo.",
            "The tune in [Key] features a pitch level of [pitch level] and a tempo described as [tempo].",
            "Playing in [Key], this melody carries a pitch that is [pitch level] and a tempo that is [tempo].",
            "This segment’s melody, in [Key], emerges with a pitch of [pitch level] and a tempo of [tempo].",
            "The melody, rooted in [Key], progresses with a [pitch level] pitch and a tempo set at [tempo].",
            "With its pitch set at [pitch level] and tempo at [tempo], this melody in [Key] captivates.",
            "Orchestrated in [Key], the melody elevates to a pitch of [pitch level] and moves at a [tempo] tempo.",
            "The song section in [Key] maintains a pitch of [pitch level] and pulses at a [tempo] beat.",
            "This tune, hailing from [Key], registers a pitch of [pitch level] and a tempo termed [tempo].",
            "The melody, nestled in the tonality of [Key], showcases a [pitch level] pitch alongside a tempo described as [tempo].",
        ]

        self.templates_wo_key_and_avg_pitch = [
            "The melody progresses at a [tempo] pace, brimming with [emotional characteristics] feel.",
            "This segment’s tempo is [tempo], conveying a distinctly [emotional characteristics] mood.",
            "With a [tempo] tempo, this melody radiates [emotional characteristics] vibes.",
            "The tune moves at a [tempo] pace, filled with [emotional characteristics] emotions.",
            "Set at a [tempo] speed, the melody pulses with a [emotional characteristics] energy.",
            "This melody, flowing at a [tempo] tempo, is infused with [emotional characteristics] characteristics.",
            "At a [tempo] tempo, this segment of the song exudes [emotional characteristics] mood.",
            "The melody’s pace of [tempo] complements its deeply [emotional characteristics] nature.",
            "This tune, with its [tempo] rhythm, showcases [emotional characteristics] tone.",
            "Moving at a [tempo] speed, the melody embodies a [emotional characteristics] aura.",
        ]

        self.templates_wo_key_and_tempo = [
            "The melody in this segment, having a [pitch level] pitch, pulsates with [emotional characteristics] energy.",
            "With its [pitch level] pitch, this melody carries a vibrancy filled with [emotional characteristics] qualities.",
            "The song's segment features a melody at a [pitch level] pitch, rich with [emotional characteristics] emotions.",
            "This melody, pitched at a [pitch level], radiates [emotional characteristics] emotions, full of energy and passion.",
            "The [pitch level] pitch of this segment's melody enhances its [emotional characteristics] essence.",
            "Characterized by a [pitch level] pitch, the melody emanates [emotional characteristics] characteristics, bursting with vigorous and assertive tones.",
            "In this music piece, the melody reaches a [pitch level] pitch and exudes an [emotional characteristics] mood, lively and forceful.",
            "This song segment, with its melody at a [pitch level] pitch, is imbued with a profoundly [emotional characteristics] atmosphere.",
            "Flowing at a [pitch level] pitch, the melody of this segment unfolds with dynamic [emotional characteristics] energy.",
            "The melody of this song part, notable for its [pitch level] pitch, carries the intense emotions of [emotional characteristics].",
        ]

        self.templates_wo_key_and_emotion = [
            "This song segment features a melody with a [pitch level] pitch and a [tempo] tempo, creating a distinctive sound.",
            "With its melody at a [pitch level] pitch and moving at a [tempo] tempo, this part of the song offers a unique listening experience.",
            "The melody in this segment rises to a [pitch level] pitch and unfolds at a [tempo] pace, delivering a balanced musical expression.",
            "This piece of music, characterized by a [pitch level] pitch and a [tempo] tempo, showcases the song's dynamic range.",
            "In this song segment, the melody achieves a [pitch level] pitch and progresses with a [tempo] tempo, setting the tone for the piece.",
            "The melody's [pitch level] pitch coupled with its [tempo] tempo in this segment gives the music a distinct flavor.",
            "A melody with a [pitch level] pitch and [tempo] tempo defines this vibrant segment of the song.",
            "This part of the song, featuring a melody at a [pitch level] pitch and moving at a [tempo] pace, resonates with clarity and rhythm.",
            "The song's segment, played at a [pitch level] pitch and a [tempo] tempo, creates a captivating musical journey.",
            "With a melody that holds a [pitch level] pitch and a tempo of [tempo], this segment of the song stands out for its expressive qualities.",
        ]

        self.templates_wo_avg_pitch_and_tempo = [
            "The melody of this song, set in [Key], carries a distinctly [emotional characteristics] aura, permeating the segment with its charm.",
            "In the key of [Key], this melody unfolds with a [emotional characteristics] atmosphere, enriching the musical experience.",
            "Crafted in [Key], the melody radiates a [emotional characteristics] mood, setting a heartfelt tone throughout the segment.",
            "This segment’s melody, tuned to [Key], encapsulates a [emotional characteristics] vibe, seamlessly blending emotion with melody.",
            "With its harmonious tones in [Key], the melody evokes a [emotional characteristics] feel, enhancing the song's emotional depth.",
            "The melody, flowing in the key of [Key], exudes a [emotional characteristics] ambiance, captivating the listener's heart.",
            "Set in [Key], this melody embodies a [emotional characteristics] essence, painting a vivid emotional landscape.",
            "Originating in [Key], the melody of this song segment imbues the air with a [emotional characteristics] spirit, weaving a tapestry of emotion.",
            "In the musical key of [Key], this melody sweeps through with a [emotional characteristics] sentiment, coloring the piece with emotional tones.",
            "This melody, resonating in [Key], carries with it a deeply [emotional characteristics] atmosphere, profoundly affecting the listener's experience.",
        ]

        self.templates_wo_avg_pitch_and_emotion = [
            "The melody of this segment is composed in [Key], moving at a [tempo] pace.",
            "In the key of [Key], this melody unfolds with a tempo that is distinctly [tempo].",
            "This piece features a melody in [Key], characterized by its [tempo] tempo.",
            "With its tempo set at [tempo], the melody in [Key] flows smoothly through the segment.",
            "The song segment, played in [Key], progresses at a [tempo] speed.",
            "Set in the tonal center of [Key], this melody's tempo maintains a [tempo] pace, providing a consistent rhythmic feel.",
            "This melody, resonating in [Key], operates at a tempo that is comfortably [tempo].",
            "The tune, anchored in [Key], moves along at a [tempo] pace, blending seamlessly with the song’s dynamics.",
            "This musical segment, crafted in [Key], is executed with a tempo that feels [tempo], adding a unique character to the melody.",
            "With a tempo marked as [tempo], the melody set in [Key] offers a rhythmic experience that defines the song’s flow.",
        ]

        self.templates_wo_tempo_and_emotion = [
            "The melody progresses at a [tempo] pace, imbued with a sense of [emotional characteristics] energy.",
            "With a [tempo] tempo, this melody exudes [emotional characteristics] moods, adding intensity to the song.",
            "The tune, moving at a [tempo] speed, is rich with [emotional characteristics] vibes, reflecting its dynamic nature.",
            "Set to a [tempo] rhythm, the melody thrives with an [emotional characteristics] aura that captivates.",
            "Flowing at a [tempo] pace, this segment's melody radiates [emotional characteristics] emotions, energizing the listener.",
            "The melody's [tempo] tempo brings out its [emotional characteristics] nature, making it memorable.",
            "Characterized by its [tempo] speed, the melody pulsates with a vivid display of [emotional characteristics] tone.",
            "This piece's melody, with its [tempo] tempo, is permeated with a [emotional characteristics] energy.",
            "At a [tempo] pace, the melody showcases a compelling [emotional characteristics] essence that enhances the song.",
            "The song unfolds at a [tempo] tempo, filled with the [emotional characteristics] feel that defines its spirit.",
        ]

        self.templates_wo_key_and_avg_pitch_and_tempo = [
            "The melody of this segment carries a [emotional characteristics] essence.",
            "Infused with [emotional characteristics] mood, the tune effectively conveys a sense of dynamism and intensity.",
            "This piece of music exudes a [emotional characteristics] aura, capturing the listener's attention with its powerful emotion.",
            "Rich in [emotional characteristics] emotions, the melody evokes a sense of vigorous and forceful expression.",
            "The emotional undertone of this melody is distinctly [emotional characteristics].",
        ]

        self.templates_wo_key_and_avg_pitch_and_emotion = [
            "The melody moves at a [tempo] pace, setting the rhythm of the segment.",
            "With a [tempo] tempo, this tune dictates the overall speed of the music.",
            "The pace of the melody is distinctly [tempo], which shapes the feel of the song.",
            "This segment’s melody flows with a [tempo] speed, aligning perfectly with the mood of the piece.",
            "Characterized by its [tempo] tempo, the melody provides a steady rhythmic backdrop.",
        ]

        self.avg_pitch_phrases = {
            'low': ['low', 'relatively low'],
            'medium': ['medium', 'average'],
            'high': ['high', 'relatively high'],
            'very high': ['very high'],
            'None': [None]
        }

        self.tempo_phrases = {
            'very low': ['very slow'],
            'low': ['slow', 'low', 'gentle'],
            'medium': ['medium', 'average', 'moderate'],
            'high': ['fast', 'rapid', 'swift', 'quick'],
            'very high': ['very fast'],
            'None': [None]
        }

        self.duration_phrases = {
            'short': ['a short period of time', 'a short period', 'a short duration'],
            'medium': ['a medium period of time'],
            'long': ['a long period of time', 'a long period', 'a long duration'],
            'very long': ['a very long period of time'],
            'None': [None]
        }

    def prepare_key(self, key, key_conf):
        if key is None or key == 'None' or key_conf < self.key_min_conf:
            key = None
        else:
            key = music21.key.Key(key)
            if np.random.random(1) > 0.5:
                key = key.relative
            possible_key_names = [
                f"{key.tonic.fullName} {key.mode}",
                f"{key.tonic.step} {key.tonic.accidental} {key.mode}",
                f"{key.name}"
            ]
            key = np.random.choice(possible_key_names)
        return key

    def prepare_tempo(self, tempo, tempo_conf):
        if tempo is None or tempo <= 0 or tempo_conf < self.tempo_min_conf:
            tempo = None
        else:
            if tempo < 70:
                tempo = 'very low'
            elif 70 <= tempo < 90:
                tempo = 'low'
            elif 90 <= tempo < 120:
                tempo = 'medium'
            elif 120 <= tempo < 160:
                tempo = 'high'
            elif 160 <= tempo:
                tempo = 'very high'
            tempo = np.random.choice(self.tempo_phrases[tempo])
        return tempo

    def prepare_avg_pitch(self, avg_pitch):
        if avg_pitch is None or avg_pitch <= 0:
            avg_pitch = None
        else:
            if avg_pitch < 56:
                avg_pitch = 'low'
            elif 56 <= avg_pitch < 63:
                avg_pitch = 'medium'
            elif 63 <= avg_pitch < 78:
                avg_pitch = 'high'
            elif 78 <= avg_pitch:
                avg_pitch = 'very high'
            avg_pitch = np.random.choice(self.avg_pitch_phrases[avg_pitch])
        return avg_pitch

    def prepare_emotion(self, emotion):
        if emotion is None or len(emotion) == 0 or emotion == 'None':
            emotion = None
        else:
            if len(emotion) == 1:
                emotion = emotion[0]
            elif len(emotion) == 2:
                emotion = ' and '.join(random.sample(emotion, len(emotion)))
            elif len(emotion) > 2:
                emotions = random.sample(emotion, len(emotion))
                emotion = ', '.join(emotions[:-1]) + ', and ' + emotions[-1]
        return emotion

    def prepare_duration(self, duration):
        if duration is None or duration <= 0:
            duration = None
        else:
            orig_duration = f"{round(duration)} seconds"
            if duration < 5:
                duration = 'short'
            elif 5 <= duration < 10:
                duration = 'medium'
            elif 10 <= duration < 15:
                duration = 'long'
            elif 15 <= duration:
                duration = 'very long'
            duration = np.random.choice(self.duration_phrases[duration])
            duration = np.random.choice([duration, orig_duration])  # a chance to provide precise durations
        return duration

    def transcribe(self, key=None, key_conf=0., avg_pitch=None, tempo=None, tempo_conf=0., emotion=None, duration=None):
        key = self.prepare_key(key, key_conf)
        tempo = self.prepare_tempo(tempo, tempo_conf)
        avg_pitch = self.prepare_avg_pitch(avg_pitch)
        emotion = self.prepare_emotion(emotion)
        duration = self.prepare_duration(duration)

        code = ''.join([str(int(key is not None)), str(int(avg_pitch is not None)), str(int(tempo is not None)), str(int(emotion is not None))])

        caption = None
        if duration is None:
            if code == '1111':
                caption = np.random.choice(self.full_factor_templates)
                caption = caption.replace(self.key_kw, key).replace(self.avg_pitch_kw, avg_pitch).replace(self.tempo_kw, tempo).replace(self.emotion_kw, emotion)
            elif code == '0111':
                caption = np.random.choice(self.templates_wo_key)
                caption = caption.replace(self.avg_pitch_kw, avg_pitch).replace(self.tempo_kw, tempo).replace(self.emotion_kw, emotion)
            elif code == '1011':
                caption = np.random.choice(self.templates_wo_avg_pitch)
                caption = caption.replace(self.key_kw, key).replace(self.tempo_kw, tempo).replace(self.emotion_kw, emotion)
            elif code == '1101':
                caption = np.random.choice(self.templates_wo_tempo)
                caption = caption.replace(self.key_kw, key).replace(self.avg_pitch_kw, avg_pitch).replace(self.emotion_kw, emotion)
            elif code == '1110':
                caption = np.random.choice(self.full_factor_templates)
                caption = caption.replace(self.key_kw, key).replace(self.avg_pitch_kw, avg_pitch).replace(self.tempo_kw, tempo)
            elif code == '0011':
                caption = np.random.choice(self.templates_wo_key_and_avg_pitch).replace(self.tempo_kw, tempo).replace(self.emotion_kw, emotion)
            elif code == '0101':
                caption = np.random.choice(self.templates_wo_key_and_tempo)
                caption = caption.replace(self.avg_pitch_kw, avg_pitch).replace(self.emotion_kw, emotion)
            elif code == '0110':
                caption = np.random.choice(self.templates_wo_key_and_emotion)
                caption = caption.replace(self.avg_pitch_kw, avg_pitch).replace(self.tempo_kw, tempo)
            elif code == '1001':
                caption = np.random.choice(self.templates_wo_avg_pitch_and_tempo)
                caption = caption.replace(self.key_kw, key).replace(self.emotion_kw, emotion)
            elif code == '1010':
                caption = np.random.choice(self.templates_wo_avg_pitch_and_emotion)
                caption = caption.replace(self.key_kw, key).replace(self.tempo_kw, tempo)
            elif code == '1100':
                caption = np.random.choice(self.templates_wo_tempo_and_emotion)
                caption = caption.replace(self.key_kw, key).replace(self.avg_pitch_kw, avg_pitch)
            elif code == '0001':
                caption = np.random.choice(self.templates_wo_key_and_avg_pitch_and_tempo).replace(self.emotion_kw, emotion)
            elif code == '0010':
                caption = np.random.choice(self.full_factor_templates).replace(self.tempo_kw, tempo)
            else:
                caption = ''
        else:
            if code == '1111':
                caption = np.random.choice(self.full_factor_templates_w_duration)
                caption = caption.replace(self.key_kw, key).replace(self.avg_pitch_kw, avg_pitch).replace(self.tempo_kw, tempo).replace(self.emotion_kw, emotion).replace(self.duration_kw, duration)
            elif code == '0111':
                caption = np.random.choice(self.templates_wo_key_w_duration)
                caption = caption.replace(self.avg_pitch_kw, avg_pitch).replace(self.tempo_kw, tempo).replace(self.emotion_kw, emotion).replace(self.duration_kw, duration)
            elif code == '1011':
                caption = np.random.choice(self.templates_wo_avg_pitch_w_duration)
                caption = caption.replace(self.key_kw, key).replace(self.tempo_kw, tempo).replace(self.emotion_kw, emotion).replace(self.duration_kw, duration)
            elif code == '1101':
                caption = np.random.choice(self.templates_wo_tempo_w_duration)
                caption = caption.replace(self.key_kw, key).replace(self.avg_pitch_kw, avg_pitch).replace(self.emotion_kw, emotion).replace(self.duration_kw, duration)
            elif code == '1110':
                caption = np.random.choice(self.full_factor_templates_w_duration)
                caption = caption.replace(self.key_kw, key).replace(self.avg_pitch_kw, avg_pitch).replace(self.tempo_kw, tempo).replace(self.duration_kw, duration)
            elif code == '0011':
                caption = np.random.choice(self.templates_wo_key_and_avg_pitch).replace(self.tempo_kw, tempo).replace(self.emotion_kw, emotion).replace(self.duration_kw, duration)
            elif code == '0101':
                caption = np.random.choice(self.templates_wo_key_and_tempo_w_duration)
                caption = caption.replace(self.avg_pitch_kw, avg_pitch).replace(self.emotion_kw, emotion).replace(self.duration_kw, duration)
            elif code == '0110':
                caption = np.random.choice(self.templates_wo_key_and_emotion_w_duration)
                caption = caption.replace(self.avg_pitch_kw, avg_pitch).replace(self.tempo_kw, tempo).replace(self.duration_kw, duration)
            elif code == '1001':
                caption = np.random.choice(self.templates_wo_avg_pitch_and_tempo_w_duration)
                caption = caption.replace(self.key_kw, key).replace(self.emotion_kw, emotion).replace(self.duration_kw, duration)
            elif code == '1010':
                caption = np.random.choice(self.templates_wo_avg_pitch_and_emotion_w_duration)
                caption = caption.replace(self.key_kw, key).replace(self.tempo_kw, tempo).replace(self.duration_kw, duration)
            elif code == '1100':
                caption = np.random.choice(self.templates_wo_tempo_and_emotion_w_duration)
                caption = caption.replace(self.key_kw, key).replace(self.avg_pitch_kw, avg_pitch).replace(self.duration_kw, duration)
            elif code == '0001':
                caption = np.random.choice(self.templates_wo_key_and_avg_pitch_and_tempo_w_duration).replace(self.emotion_kw, emotion).replace(self.duration_kw, duration)
            elif code == '0010':
                caption = np.random.choice(self.full_factor_templates_w_duration).replace(self.tempo_kw, tempo).replace(self.duration_kw, duration)
            else:
                caption = ''

        return caption


class CaptionGenerator2(CaptionGenerator):
    """
    在分割几个描述词的时候，将分割线附近的特征故意忽略，避免让模型困惑。中间的模糊地带直接None
    """
    def prepare_tempo(self, tempo, tempo_conf):
        if tempo is None or tempo <= 0 or tempo_conf < self.tempo_min_conf:
            tempo = None
        else:
            if tempo < 69:
                tempo = 'very low'
            elif 71 <= tempo < 89:
                tempo = 'low'
            elif 91 <= tempo < 119:
                tempo = 'medium'
            elif 121 <= tempo < 159:
                tempo = 'high'
            elif 161 <= tempo:
                tempo = 'very high'
            else:
                tempo = 'None'
            tempo = np.random.choice(self.tempo_phrases[tempo])
        return tempo

    def prepare_avg_pitch(self, avg_pitch):
        if avg_pitch is None or avg_pitch <= 0:
            avg_pitch = None
        else:
            if avg_pitch < 53:
                avg_pitch = 'low'
            elif 56 <= avg_pitch < 62:
                avg_pitch = 'medium'
            elif 64 <= avg_pitch < 77:
                avg_pitch = 'high'
            elif 79 <= avg_pitch:
                avg_pitch = 'very high'
            else:
                avg_pitch = 'None'
            avg_pitch = np.random.choice(self.avg_pitch_phrases[avg_pitch])
        return avg_pitch

    def prepare_duration(self, duration):
        if duration is None or duration <= 0:
            duration = None
        else:
            orig_duration = f"{round(duration)} seconds"
            if duration < 4.5:
                duration = 'short'
            elif 5.5 <= duration < 9.5:
                duration = 'medium'
            elif 10.5 <= duration < 14.5:
                duration = 'long'
            elif 15.5 <= duration:
                duration = 'very long'
            else:
                duration = 'None'
            duration = np.random.choice(self.duration_phrases[duration])
            duration = np.random.choice([duration, orig_duration])  # a chance to provide precise durations
        return duration


# %%
# manifest_path = '/mnt/sdb/liruiqi/datasets/TTSong/music_feats/crawl_new_music_feat.tsv'
#
# items = load_samples_from_tsv(manifest_path)
#
# transcriber = MusicFeatToCaption()
#
# new_items = []
# for item in tqdm(items):
#     caption = transcriber.transcribe(
#         key=item['key'],
#         key_conf=float(item['key_confidence']),
#         avg_pitch=float(item['avg_pitch']),
#         tempo=float(item['tempo']),
#         tempo_conf=float(item['tempo_confidence']),
#         emotion=eval(item['emotion']),
#         duration=float(item['duration'])
#     )
#     new_items.append({
#         'item_name': item['item_name'],
#         'caption': caption
#     })
