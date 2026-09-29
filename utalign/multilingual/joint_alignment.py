"""Joint MIDI assignment using acoustic phone events, then constrained full CTC.

Every lyric syllable is assigned. MIDI notes may be shared, sustained across, or
skipped; no lyric-deletion transition exists. Impossible inputs fail explicitly.
"""
import math
import numpy as np
from numba import njit


@njit(cache=True)
def event_cost(em, targets, cuts, nuclei, i, k, start, end, lead, tail):
    """Ordered acoustic phone evidence in a proposed MIDI interval.

    This assignment cost locates ordered phone events, not their duration.
    The subsequent full CTC pass supplies actual frame spans and blank states.
    """
    a=max(0,int(math.floor((start-lead)/.02)))
    b=min(len(em),int(math.ceil((end+tail)/.02)))
    left=cuts[i];right=cuts[i+k];count=right-left
    if b-a<count or count==0:
        return 1e20
    previous=np.zeros(b-a)
    for q in range(left,right):
        current=np.full(b-a,-1e20)
        running=-1e20
        gap=2 if q>left and targets[q]==targets[q-1] else 1
        owner=i
        while q>=cuts[owner+1]:
            owner+=1
        is_nucleus=q==nuclei[owner]
        expected=start+(owner-i)*max(.02,end-start)/k
        for t in range(b-a):
            if q==left:
                running=0.
            elif t-gap>=0:
                running=max(running,previous[t-gap])
            penalty=.65*min(2.,abs((a+t)*.02-expected)) if is_nucleus else 0.
            current[t]=running+em[a+t,targets[q]]-penalty
        previous=current
    best=np.max(previous)
    if best < -1e19:
        return 1e20
    return -best/count*k


@njit(cache=True)
def assign_notes(em, targets, cuts, nuclei, on, off, raw, voiced,
                 max_shared=3, max_melisma=8, lead=.14, tail=.10, band=10.):
    m=len(cuts)-1;n=len(on)
    cost=np.full((m+1,n+1),1e20)
    back_k=np.zeros((m+1,n+1),np.int16);back_l=np.zeros((m+1,n+1),np.int16)
    cost[0,0]=0.
    for i in range(m+1):
        for j in range(n+1):
            base=cost[i,j]
            if base>=1e19:
                continue
            if j<n:
                skip=2.8 if voiced[j] else .2
                if base+skip<cost[i,j+1]:
                    cost[i,j+1]=base+skip;back_k[i,j+1]=0;back_l[i,j+1]=1
            if i>=m or j>=n:
                continue
            if math.isfinite(raw[i]) and abs(raw[i]-on[j])>band:
                continue
            anchor=.10*min(8.,abs(raw[i]-on[j])) if math.isfinite(raw[i]) else 0.
            for k in range(1,min(max_shared,m-i)+1):
                acoustic=event_cost(em,targets,cuts,nuclei,i,k,on[j],off[j],lead,tail)
                short=max(0.,.07*k-(off[j]-on[j]))*5
                candidate=base+acoustic+.35*(k-1)+short+anchor
                if candidate<cost[i+k,j+1]:
                    cost[i+k,j+1]=candidate;back_k[i+k,j+1]=k;back_l[i+k,j+1]=1
            for length in range(2,min(max_melisma,n-j)+1):
                # A long rest terminates a melisma, instead of spanning entire verses.
                if on[j+length-1]-off[j+length-2]>.5:
                    break
                acoustic=event_cost(em,targets,cuts,nuclei,i,1,on[j],off[j+length-1],lead,tail)
                candidate=base+acoustic+.40*(length-1)+anchor
                if candidate<cost[i+1,j+length]:
                    cost[i+1,j+length]=candidate;back_k[i+1,j+length]=1;back_l[i+1,j+length]=length
    first=np.full(m,-1,np.int32);last=np.full(m,-1,np.int32)
    share_index=np.zeros(m,np.int32);share_count=np.ones(m,np.int32)
    if cost[m,n]>=1e19:
        return first,last,share_index,share_count,cost[m,n]
    i=m;j=n
    while i>0 or j>0:
        k=int(back_k[i,j]);length=int(back_l[i,j])
        if length==0:
            return first,last,share_index,share_count,1e20
        for q in range(k):
            first[i-k+q]=j-length;last[i-k+q]=j-1
            share_index[i-k+q]=q;share_count[i-k+q]=k
        i-=k;j-=length
    return first,last,share_index,share_count,cost[m,n]


@njit(cache=True)
def constrained_path(em, targets, blank, lower, upper):
    """Full CTC Viterbi; each phoneme can emit only within its assigned MIDI window."""
    tmax=len(em);smax=2*len(targets)+1
    previous=np.full(smax,-1e20);previous[0]=0.
    back=np.zeros((tmax,smax),np.uint8)
    for t in range(tmax):
        current=np.full(smax,-1e20)
        for s in range(min(smax,2*t+2)):
            token=blank if s%2==0 else targets[s//2]
            if s%2 and (t<lower[s//2] or t>=upper[s//2]):
                continue
            value=previous[s];step=0
            if s>0 and previous[s-1]>value:
                value=previous[s-1];step=1
            if s%2 and s>1 and targets[s//2]!=targets[s//2-1] and previous[s-2]>value:
                value=previous[s-2];step=2
            current[s]=value+em[t,token];back[t,s]=step
        previous=current
    state=smax-1 if previous[-1]>=previous[-2] else smax-2
    score=previous[state]
    path=np.empty(tmax,np.int32)
    if score<-1e19:
        return path,score
    for t in range(tmax-1,-1,-1):
        path[t]=state
        state-=back[t,state]
    return path,score


def align_syllables(em, document, notes, offset, aligner, wav=None, band=10.):
    import torch
    import torchaudio.functional as F
    from utalign.timing import note_voiced_flags
    syllables=document['syllables']
    cuts=np.array([0]+list(np.cumsum([len(s['model_phones']) for s in syllables])),dtype=np.int32)
    targets=np.array([aligner.vocab[p] for s in syllables for p in s['model_phones']],dtype=np.int32)
    nuclei=np.array([cuts[i]+s['nucleus_token'] for i,s in enumerate(syllables)],dtype=np.int32)
    # This unconstrained pass supplies only a weak search prior. It is not clamped into MIDI times.
    raw_path,raw_scores=F.forced_align(torch.from_numpy(np.ascontiguousarray(em))[None],
                                     torch.from_numpy(targets)[None],blank=aligner.blank)
    raw_spans=F.merge_tokens(raw_path[0],raw_scores[0].exp(),blank=aligner.blank)
    if len(raw_spans)!=len(targets):
        raise ValueError('Initial CTC did not cover every phoneme')
    raw=np.array([raw_spans[cuts[i]].start*.02 for i in range(len(syllables))])
    on=np.array([n.start+offset for n in notes]);off=np.array([n.end+offset for n in notes])
    if np.any(off<=on) or np.any(np.diff(on)<0):
        raise ValueError('MIDI intervals must be ordered and positive')
    voiced=note_voiced_flags(wav,notes,offset) if wav is not None and np.any(wav) else np.ones(len(notes),bool)
    first,last,indices,counts,cost=assign_notes(em,targets,cuts,nuclei,on,off,raw,voiced,band=band)
    if cost>=1e19 or np.any(first<0):
        raise ValueError('No MIDI path covers every syllable; check vocal track, lyrics, offset or search band')
    lower=np.empty(len(targets),np.int32);upper=np.empty(len(targets),np.int32)
    for i in range(len(syllables)):
        lower[cuts[i]:cuts[i+1]]=max(0,int(math.floor((on[first[i]]-.14)/.02)))
        upper[cuts[i]:cuts[i+1]]=min(len(em),int(math.ceil((off[last[i]]+.10)/.02)))
    path,path_score=constrained_path(em,targets,aligner.blank,lower,upper)
    if path_score<-1e19:
        raise ValueError('MIDI assignment cannot accommodate the complete CTC phone sequence; no unconstrained fallback was exported')
    result=[]
    for i,s in enumerate(syllables):
        spans=[]
        for q in range(cuts[i],cuts[i+1]):
            frames=np.flatnonzero(path==2*q+1)
            if not len(frames):
                raise ValueError('Constrained CTC omitted a phoneme')
            spans.append(dict(phone=s['model_phones'][q-cuts[i]],start=float(frames[0]*.02),
                              end=float((frames[-1]+1)*.02),support=float(np.exp(em[frames,targets[q]].mean()))))
        support=float(np.exp(np.log(np.maximum([p['support'] for p in spans],1e-12)).mean()))
        start=spans[0]['start'];nucleus_start=spans[s['nucleus_token']]['start']
        reasons=[]
        if support<.02:reasons.append('weak_acoustic_support')
        if abs(start-raw[i])>.15:reasons.append('midi_changed_unconstrained_alignment')
        if abs(start-lower[cuts[i]]*.02)<.021:reasons.append('at_midi_window_boundary')
        result.append(dict(s,start=start,acoustic_end=spans[-1]['end'],nucleus_start=nucleus_start,
                           end=max(spans[-1]['end'],float(off[last[i]])),phone_spans=spans,
                           note_ids=list(range(int(first[i]),int(last[i])+1)),
                           share_index=int(indices[i]),share_count=int(counts[i]),
                           midi_start=float(on[first[i]]),midi_end=float(off[last[i]]),
                           raw_start=float(raw[i]),acoustic_support=support,
                           status='low_support' if support<.02 else 'aligned',diagnostics=reasons,
                           timing_basis='midi_window_constrained_ctc',end_basis='ctc_and_midi_sustain'))
    for current,nxt in zip(result,result[1:]):
        current['end']=min(current['end'],nxt['start'])
    if any(s['end']<=s['start'] for s in result):
        raise ValueError('Nonpositive syllable interval')
    used={n for s in result for n in s['note_ids']}
    summary=dict(syllables=len(result),assigned_syllables=len(result),omitted_syllables=0,
                 low_support=sum(s['status']=='low_support' for s in result),
                 shared_syllables=sum(s['share_count']>1 for s in result),
                 melisma_syllables=sum(len(s['note_ids'])>1 for s in result),
                 unused_notes=[i for i in range(len(notes)) if i not in used],
                 midi_assignment_cost=float(cost),constrained_ctc_score=float(path_score),
                 raw_shift_over150=sum(abs(s['start']-s['raw_start'])>.15 for s in result))
    return result,summary
