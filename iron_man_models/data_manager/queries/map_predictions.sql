WITH match_pred as (
select
	fm.id,
	e.name event_name,
    fm.hltv_id,
    fm.max_maps,
    fm.start_date,
	t.name as team_name,
	LOWER(m.name) as map_name,
	p.win_chance
from predictions p 
left join teams t on t.id = p.team_id
left join maps m on m.id = p.map_id
left join future_matches fm on fm.id = p.future_match_id
left join events e on e.id = fm.event_id 
),
resumo as (
SELECT 
	mp.id,
    mp.hltv_id,
    mp.max_maps,
    mp.start_date,
    mp.event_name,
	mp.map_name,
	mp.team_name,
	(mp.win_chance/(mp.win_chance + mp2.win_chance)) AS win_chance,
	ROUND(1/(mp.win_chance/(mp.win_chance + mp2.win_chance)), 3) adjusted_min_odd
FROM match_pred mp
left join match_pred mp2 on mp.id = mp2.id and mp.map_name = mp2.map_name and mp.team_name <> mp2.team_name
where ROUND(1/(mp.win_chance/(mp.win_chance + mp2.win_chance)), 3) is not null
ORDER BY id, mp.team_name, adjusted_min_odd
)
select * from resumo
