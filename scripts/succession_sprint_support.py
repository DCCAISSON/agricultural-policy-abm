"""Select complete trainable openings before examining their rewards."""

from collections import Counter
from math import ceil


def select(screen: list[tuple], previous: set[tuple] = frozenset()) -> set[tuple]:
    """Cap at ten distinct parent paths per public state; never drop prior selections."""
    by_parent = {(seed,opening):key for seed,opening,key,*_ in screen}
    bags = Counter(by_parent[parent] for parent in previous)
    new = set(previous)
    for seed,opening,key,*_ in sorted(screen):
        parent=seed,opening
        if parent not in new and bags[key]<10:
            new.add(parent)
            bags[key]+=1
    return new


def support(screen: list[tuple], selected: set[tuple], actions: int, design: dict) -> dict:
    frequencies=Counter(key for _,_,key,*_ in screen)
    years=Counter(key[0] for _,_,key,*_ in screen)
    counts=Counter(key for seed,opening,key,*_ in screen if (seed,opening) in selected)
    common={key for key,n in frequencies.items() if n>=ceil(years[key[0]]*design["common_fraction"])}
    enough={key for key,n in counts.items() if n>=design["minimum_openings"]}
    return dict(screened_openings=len(screen),opening_visits=len(selected),distinct_states=len(frequencies),
                common_state_actions=len(common)*actions,supported_common_state_actions=len(common&enough)*actions,
                all_visited_action_support=len(enough)/len(frequencies),coverage=len(common&enough)/len(common),
                opening_weighted_support=sum(frequencies[k] for k in enough)/len(screen),
                by_remaining_year=[
                    dict(remaining=y,screened=years[y],common=sum(k[0]==y for k in common),
                         supported_common=sum(k[0]==y for k in common&enough),
                         all_states=sum(k[0]==y for k in frequencies))
                    for y in sorted(years,reverse=True)],
                common_frequency=design["common_fraction"])
