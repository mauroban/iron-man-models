WITH match_pred as (
select
	fm.id,
	t.name as team_name,
	m.name as map_name,
	p.win_chance
from predictions p 
left join teams t on t.id = p.team_id
left join maps m on m.id = p.map_id
left join future_matches fm on fm.id = p.future_match_id 
where fm.hltv_id = 2376855
-- and m.name in ('Mirage', 'Dust2', 'Inferno')
)
SELECT 
	mp.id,
	mp.map_name,
	mp.team_name,
	mp2.team_name op_name,
-- 	ROUND(mp.win_chance + mp2.win_chance, 2) win_chance_sum,
	ROUND(1/(mp.win_chance/(mp.win_chance + mp2.win_chance)), 3) adjusted_min_odd
FROM match_pred mp
left join match_pred mp2 on mp.id = mp2.id and mp.map_name = mp2.map_name and mp.team_name <> mp2.team_name
ORDER BY mp.team_name, adjusted_min_odd
