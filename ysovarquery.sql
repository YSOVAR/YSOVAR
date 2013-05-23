-- standard query to get all relevant data for a cluster

SELECT source.id as ysovarid, source.name as sname, source.radeg as ra, source.dedeg as de, hmjd,filter.name as fname, mag1,emag1 ,useful,kind,photom.note from  photom,source,filter,cluster WHERE photom.source_id=source.id AND source.cluster=cluster.id AND cluster.short_name='L1688' AND photom.filter=filter.id and photom.kind = 0 AND (filter.name='IRAC1' OR filter.name='IRAC2')  AND useful= 1 AND hmjd > 55000




-- hmjd > 55000  --> excludes cold mission data
-- cluster.short_name='L1688' --> selects cluster name
-- photom.kind = 0  --> only mapping mode
-- useful = 1  --> flage that ultimately goes back to Rob...


/*
#################################################
Dear Moritz,

Because you are using four tables in the query, to uniquely match sources across 4 tables, you have to use at least 3 conditions to tie entries (N tables --> N-1 conditions). Otherwise, mysql silently "duplicate" information in its preparation for a query result. This behavior appears to be unacceptable, but in some other areas (like finance), this behavior is necessary.

So, you need to add one more condition "photom.filter=filter.id" as one of the first conditions.

SELECT source.name as sname,hmjd,filter.name as fname, mag1,emag1 ,useful,kind,photom.note from  photom,source,filter,cluster WHERE photom.source_id=source.id AND source.cluster=cluster.id AND cluster.short_name='L1688' AND photom.filter=filter.id and photom.kind = 0 AND (filter.name='IRAC1' OR filter.name='IRAC2')  AND useful= 1 and photom.source_id=215233;

Cheers,

Inseok
*/