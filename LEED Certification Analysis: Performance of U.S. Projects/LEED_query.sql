-- Create the Owners table
CREATE TABLE Owners (
    OwnerID INT PRIMARY KEY IDENTITY(1,1),
    OwnerTypes VARCHAR(MAX),
    OwnerOrganization NVARCHAR(MAX)
);

-- Create the Projects table
CREATE TABLE Projects (
    ProjectID INT PRIMARY KEY IDENTITY(1,1),
    ID VARCHAR(10) UNIQUE,
    ProjectName NVARCHAR(MAX),
    City NVARCHAR(MAX),
    State NVARCHAR(MAX),
    ZipCode VARCHAR(MAX),
    Country VARCHAR(MAX),
    GrossFloorArea FLOAT,
    TotalPropArea FLOAT,
	UnitOfMeasurement VARCHAR(MAX),
    ProjectTypes VARCHAR(MAX), -- Max len of the rows were calculated as 341
    OwnerID INT,
	IsAreaInconsistent BIT
    FOREIGN KEY (OwnerID) REFERENCES Owners(OwnerID)
);

-- Create the Certifications table
CREATE TABLE Certifications (
    CertificationID INT PRIMARY KEY IDENTITY(1,1),
    ProjectID INT,
    LeedSystemVersionDisplayName VARCHAR(MAX),
	LeedVersion VARCHAR(10)
    PointsAchieved FLOAT,
    CertLevel VARCHAR(MAX),
	RegistrationDate DATE,
    CertDate DATE,
    CertificationTimeline INT,
    IsCertified BIT,
	IsDateSwapped BIT
    FOREIGN KEY (ProjectID) REFERENCES Projects(ProjectID)
);

-- Optional: Create the Confidentiality table
CREATE TABLE Confidentiality (
    ConfidentialityID INT PRIMARY KEY IDENTITY(1,1),
    ProjectID INT,
    IsConfidential BIT,
    FOREIGN KEY (ProjectID) REFERENCES Projects(ProjectID)
);
--Schema is done, now its time to create a staging table, then import the csv at hand, than map the columns

DROP TABLE IF EXISTS staging;

CREATE TABLE staging (
	ID VARCHAR(10) UNIQUE,
	IsConfidential VARCHAR(MAX),
    ProjectName NVARCHAR(MAX),
	Street NVARCHAR(MAX),
    City NVARCHAR(MAX),
    State NVARCHAR(MAX),
    ZipCode VARCHAR(MAX),
    Country VARCHAR(MAX),
    LeedSystemVersionDisplayName VARCHAR(MAX),
    PointsAchieved VARCHAR(MAX),
    CertLevel VARCHAR(MAX),
    CertDate VARCHAR(MAX),
    IsCertified VARCHAR(MAX),
    OwnerTypes VARCHAR(MAX),
    GrossFloorArea VARCHAR(MAX),
	UnitOfMeasurement VARCHAR(MAX),
    TotalPropArea VARCHAR(MAX),
    ProjectTypes VARCHAR(MAX),
    OwnerOrganization NVARCHAR(MAX),
    RegistrationDate VARCHAR(MAX)
	LeedVersion VARCHAR(10),
	IsDateSwapped BIT,
	IsAreaInconsistent BIT
	CertificationTimeline INT
	);


-- Inserting the CSV
BULK INSERT staging
FROM 'C:\Users\MONSTER\Desktop\LEED project\PublicLEEDProjectsDirectory_cvs_utf8_repl.csv'
WITH (
    FIELDTERMINATOR = ';',
    ROWTERMINATOR = '\n',
    FIRSTROW = 2
);
-- Checking a random row if things went orderly or not
 select * from staging where ID = '1000031862';
 select top(10) * from staging order by ID


 --1.DATA VALIDATION


-- Converting IsConfidential and IsCertified
UPDATE staging
SET IsConfidential = CASE WHEN IsConfidential = 'Yes' THEN 1
					 ELSE 0 END;

UPDATE staging
SET IsCertified = CASE WHEN IsCertified = 'Yes' THEN 1
				  ELSE 0 END;

-- Example to convert CertLevel into ordinal values
UPDATE staging
SET CertLevel = CASE
	WHEN CertLevel = 'Denied' THEN 0
    WHEN CertLevel = 'Certified' THEN 1
	WHEN CertLevel = 'Bronze' THEN 2
    WHEN CertLevel = 'Silver' THEN 3
    WHEN CertLevel = 'Gold' THEN 4
    WHEN CertLevel = 'Platinum' THEN 5
    ELSE NULL -- For uncategorized or null values
END;

-- Convert PointsAchieved to FLOAT
UPDATE staging
SET PointsAchieved = ROUND(CAST(PointsAchieved AS FLOAT), 1); 

--Checking the operation
SELECT * FROM staging WHERE ISNUMERIC(PointsAchieved) = 0 AND PointsAchieved IS NOT NULL;

-- Handle invalid dates
UPDATE staging
SET CertDate = NULL
WHERE CertDate IN ('0000-00-00 00:00:00', '');

UPDATE staging
SET RegistrationDate = NULL
WHERE RegistrationDate IN ('0000-00-00 00:00:00', '');

-- Convert CertDate and RegistrationDate to DATE format
UPDATE staging
SET CertDate = 
    CASE 
        WHEN ISDATE(SUBSTRING(CertDate, 7, 4) + '-' + SUBSTRING(CertDate, 4, 2) + '-' + SUBSTRING(CertDate, 1, 2)) = 1 
        THEN CAST(SUBSTRING(CertDate, 7, 4) + '-' + SUBSTRING(CertDate, 4, 2) + '-' + SUBSTRING(CertDate, 1, 2) AS DATE)
        ELSE NULL 
    END
WHERE CertDate IS NOT NULL;

UPDATE staging
SET RegistrationDate = 
    CASE 
        WHEN ISDATE(SUBSTRING(RegistrationDate, 7, 4) + '-' + SUBSTRING(RegistrationDate, 4, 2) + '-' + SUBSTRING(RegistrationDate, 1, 2)) = 1 
        THEN CAST(SUBSTRING(RegistrationDate, 7, 4) + '-' + SUBSTRING(RegistrationDate, 4, 2) + '-' + SUBSTRING(RegistrationDate, 1, 2) AS DATE)
        ELSE NULL 
    END
WHERE RegistrationDate IS NOT NULL;
-- Convert GrossFloorArea and TotalPropArea to FLOAT
UPDATE staging
SET GrossFloorArea = CAST(GrossFloorArea AS FLOAT),  
    TotalPropArea = CAST(TotalPropArea AS FLOAT);
-- Set the values with 0 as NULLS from area columns
UPDATE staging
SET GrossFloorArea = NULL
WHERE CAST(GrossFloorArea AS FLOAT) = 0;

UPDATE staging
SET TotalPropArea = NULL
WHERE CAST(TotalPropArea AS FLOAT) = 0;

-- Removing all the empty ones, labeling them as nulls
	-- Update empty strings to NULL for ProjectName column
	UPDATE staging
	SET ProjectName = NULL
	WHERE ProjectName = '';

	-- Update empty strings to NULL for City column
	UPDATE staging
	SET City = NULL
	WHERE City = '';

	UPDATE staging
	SET Street = NULL
	WHERE Street = '';

	-- Update empty strings to NULL for State column
	UPDATE staging
	SET State = NULL
	WHERE State = '';

	-- Update empty strings to NULL for ZipCode column
	UPDATE staging
	SET ZipCode = NULL
	WHERE ZipCode = '';

	-- Update empty strings to NULL for Country column
	UPDATE staging
	SET Country = NULL
	WHERE Country = '';

	-- Update empty strings to NULL for LeedSystemVersionDisplayName column
	UPDATE staging
	SET LeedSystemVersionDisplayName = NULL
	WHERE LeedSystemVersionDisplayName = '';

	-- Update empty strings to NULL for PointsAchieved column
	UPDATE staging
	SET PointsAchieved = NULL
	WHERE PointsAchieved = '';

	-- Update empty strings to NULL for CertLevel column
	UPDATE staging
	SET CertLevel = NULL
	WHERE CertLevel = '';

	-- Update empty strings to NULL for CertDate column
	UPDATE staging
	SET CertDate = NULL
	WHERE CertDate = '';

	-- Update empty strings to NULL for IsCertified column
	UPDATE staging
	SET IsCertified = NULL
	WHERE IsCertified = '';

	-- Update empty strings to NULL for OwnerTypes column
	UPDATE staging
	SET OwnerTypes = NULL
	WHERE OwnerTypes = '';

	-- Update empty strings to NULL for GrossFloorArea column
	UPDATE staging
	SET GrossFloorArea = NULL
	WHERE GrossFloorArea = '';

	-- Update empty strings to NULL for UnitOfMeasurement column
	UPDATE staging
	SET UnitOfMeasurement = NULL
	WHERE UnitOfMeasurement = '';

	-- Update empty strings to NULL for TotalPropArea column
	UPDATE staging
	SET TotalPropArea = NULL
	WHERE TotalPropArea = '';

	-- Update empty strings to NULL for ProjectTypes column
	UPDATE staging
	SET ProjectTypes = NULL
	WHERE ProjectTypes = '';

	-- Update empty strings to NULL for OwnerOrganization column
	UPDATE staging
	SET OwnerOrganization = NULL
	WHERE OwnerOrganization = '';

	-- Update empty strings to NULL for RegistrationDate column
	UPDATE staging
	SET RegistrationDate = NULL
	WHERE RegistrationDate = '';

BEGIN TRANSACTION;

-- Convert PointsAchieved to FLOAT
ALTER TABLE staging
ALTER COLUMN PointsAchieved FLOAT;

-- Convert CertDate and RegistrationDate to DATE format
ALTER TABLE staging
ALTER COLUMN CertDate DATE;

ALTER TABLE staging
ALTER COLUMN RegistrationDate DATE;

-- Convert GrossFloorArea and TotalPropArea to FLOAT
ALTER TABLE staging
ALTER COLUMN GrossFloorArea FLOAT;

ALTER TABLE staging
ALTER COLUMN TotalPropArea FLOAT;

ALTER TABLE staging
ALTER COLUMN IsConfidential BIT;

ALTER TABLE staging
ALTER COLUMN IsCertified BIT;

COMMIT;

--------------------------------------------------------------------------
--2.CLEANING
select * from staging 

-- Checking and deleting ID duplicates
DELETE FROM staging
WHERE ID IN (
    SELECT ID
    FROM (
        SELECT ID, ROW_NUMBER() OVER (PARTITION BY ID ORDER BY ID) AS rn
        FROM staging
    ) AS duplicates
    WHERE rn > 1
);

-- Identify potential duplicates by ProjectName
SELECT ProjectName, COUNT(*) AS NumOccurrences
FROM staging
GROUP BY ProjectName
HAVING COUNT(*) > 1;
-- Remove leading and trailing spaces
UPDATE staging
SET ProjectName = TRIM(ProjectName);

UPDATE staging
SET Street = TRIM(Street);

-- Delete rows where LeedSystemVersionDisplayName does not start with 'LEED'
DELETE FROM staging
WHERE LeedSystemVersionDisplayName NOT LIKE 'LEED%';
-- Delete rows that shows irrational numbers regarding the LEED Pointification
BEGIN TRANSACTION; 

DELETE FROM staging 
WHERE PointsAchieved > 130;

SELECT DISTINCT PointsAchieved FROM staging ORDER BY PointsAchieved;

COMMIT;

-- Update points from 0 to 20 where the project is certified but has no points recorded. From the observation,they seem to be the floor value of 20 but didnt registered.
UPDATE staging
SET PointsAchieved = 20
WHERE PointsAchieved = 0 
AND IsCertified = 1
AND CertLevel IS NOT NULL;

BEGIN TRANSACTION;
-- New column for version specification
ALTER TABLE staging ADD LeedVersion VARCHAR(10);

UPDATE staging
SET LeedVersion = CASE
    WHEN LeedSystemVersionDisplayName LIKE '%v4%' THEN 'v4'
    WHEN LeedSystemVersionDisplayName LIKE '%1.%' THEN 'v1'
    WHEN LeedSystemVersionDisplayName LIKE '%2.%' THEN 'v2'
    WHEN LeedSystemVersionDisplayName LIKE '%20%' THEN 'v3'
    WHEN LeedSystemVersionDisplayName LIKE '%O&M%' THEN 'v2'
    -- Treat these text-based descriptions as 'v4'
    WHEN LeedSystemVersionDisplayName IN ('LEED for Cities', 
                                          'LEED for Communities', 
                                          'LEED For Homes Multi Family Low-Rise') THEN 'v4'
    -- Treat these text-based descriptions as 'v3'
    WHEN LeedSystemVersionDisplayName IN ('LEED for Retail (CI) Pilot', 
                                          'LEED for Retail (NC)', 
                                          'LEED for Retail (New Construction) Pilot',
										  'LEED for Schools',
										  'LEED Master Site BD+C') THEN 'v3'
    ELSE 'Other'  -- Optional: catch-all for any unrecognized versions
END;


SELECT LeedSystemVersionDisplayName, LeedVersion
FROM staging
GROUP BY LeedSystemVersionDisplayName, LeedVersion
ORDER BY LeedVersion;

COMMIT;

-----------------------------------------------------------------------------------------
-- Query to analyze point binning and how it corresponds to CertLevel across LEED versions to decide on cleaning
-- COMMENT: V4's are more or less consistent but v2's differing a bit, both binning and versions, some may have been binned in different pointification
WITH PointsBinned AS (
    SELECT 
        CASE 
            WHEN LeedSystemVersionDisplayName LIKE '%v4%' THEN 'Version 4'
            WHEN LeedSystemVersionDisplayName LIKE '%v2%' OR LeedSystemVersionDisplayName LIKE '%2.%' THEN 'Version 2'
            ELSE 'Other'
        END AS VersionCategory,
        CertLevel,
        PointsAchieved,
        CASE 
            WHEN PointsAchieved BETWEEN 40 AND 49 THEN '40-49'
            WHEN PointsAchieved BETWEEN 50 AND 59 THEN '50-59'
            WHEN PointsAchieved BETWEEN 60 AND 79 THEN '60-79'
            WHEN PointsAchieved >= 80 THEN '80+'
            ELSE 'Below 40'
        END AS PointsBin
    FROM 
        staging
    WHERE 
        LeedSystemVersionDisplayName LIKE '%v4%' 
        OR LeedSystemVersionDisplayName LIKE '%v2%' 
        OR LeedSystemVersionDisplayName LIKE '%2.%'
)

SELECT 
    VersionCategory,
    PointsBin,
    CertLevel,
    COUNT(*) AS NumProjects,
    ROUND(CAST(COUNT(*) AS FLOAT) * 100 / SUM(COUNT(*)) OVER (PARTITION BY VersionCategory, PointsBin), 2) AS Percentage
FROM 
    PointsBinned
GROUP BY 
    VersionCategory,
    PointsBin,
    CertLevel
ORDER BY 
    VersionCategory, PointsBin, CertLevel;

----------------------------------------------------------------------------------------
--Swapping the registration and certification dates for the rows showing inconsistency and creating flag column, noting the operation
--NOTE: With the inspection of the alleged rows, ruled them as data misplacement error, but also created a flag column to keep them in track.
BEGIN TRANSACTION;

ALTER TABLE staging ADD IsDateSwapped BIT;

UPDATE staging
SET 
    RegistrationDate = CertDate,
    CertDate = RegistrationDate,
    IsDateSwapped = 1
WHERE 
    RegistrationDate > CertDate;

SELECT *
FROM staging
WHERE IsDateSwapped = 1;

UPDATE staging
SET IsDateSwapped = 0
WHERE IsDateSwapped IS NULL

COMMIT;

select * from staging

select * from staging where OwnerTypes IN ('University')

--Tidying up the OwnerTypes, converging typos into one coherent expression

BEGIN TRANSACTION;

-- Update Community Development Corporation
UPDATE staging
SET OwnerTypes = 'Community Development Corporation or Non-Profit'
WHERE OwnerTypes IN ('Community Development Corporation o', 'Community Development Corporation or Non');

-- Update Corporate: Publicly Traded
UPDATE staging
SET OwnerTypes = 'Corporate: Publicly Traded'
WHERE OwnerTypes IN ('Corporate: Publicly  Traded', 'Corporate: Publicly Traded', 'Corporate:Publicly Traded');

-- Update Educational: Community College, Private
UPDATE staging
SET OwnerTypes = 'Educational: Community College, Private'
WHERE OwnerTypes = 'Educational: Community College, Pri';

-- Update Educational: Community College, Public
UPDATE staging
SET OwnerTypes = 'Educational: Community College, Public'
WHERE OwnerTypes = 'Educational: Community College, Pub';

-- Update Educational: Early Childhood Education
UPDATE staging
SET OwnerTypes = 'Educational: Early Childhood Education'
WHERE OwnerTypes IN ('Educational: Early Childhood Educat', 'Educational: Early Childhood Education/D');

-- Update Government Use: Local, Public Housing Authority
UPDATE staging
SET OwnerTypes = 'Government Use: Local, Public Housing Authority'
WHERE OwnerTypes IN ('Government Use: Local, Public Housi', 'Government Use: Local, Public Housing Au', 'Government Use: Local, Publlic Hous');

-- Update Government Use: Other (utility, airport)
UPDATE staging
SET OwnerTypes = 'Government Use: Other (utility, airport)'
WHERE OwnerTypes IN ('Government Use: Other (utility airp', 'Government Use: Other (utility, air', 'Government Use: Other (utility, airport,');

-- Update Investor: REIT, Non-traded
UPDATE staging
SET OwnerTypes = 'Investor: REIT, Non-traded'
WHERE OwnerTypes IN ('Investor: REIT, Non-traded', 'Investor: REIT,Non-traded', 'Investor: REIT-Non-traded', 'Investor:REIT,Non-traded');

-- Update Investor: REIT, Publicly traded
UPDATE staging
SET OwnerTypes = 'Investor: REIT, Publicly traded'
WHERE OwnerTypes IN ('Investor: REIT, Publicly traded', 'Investor: REIT-Publicly traded');

-- Update Local Government (municipalities and counties)
UPDATE staging
SET OwnerTypes = 'Local Government (municipalities and counties)'
WHERE OwnerTypes LIKE 'Local Government (municipalities an%';

-- Update Non-Profit (that do not fit into other categories)
UPDATE staging
SET OwnerTypes = 'Non-Profit (that do not fit into other categories)'
WHERE OwnerTypes LIKE 'Non-Profit (that do not fit into ot%' OR OwnerTypes LIKE 'Non-Profit (that does not fit into%';

-- Update Other categories to change word order
UPDATE staging
SET OwnerTypes = 'Individual, Other'
WHERE OwnerTypes = 'Other, Individual';

UPDATE staging
SET OwnerTypes = 'Local Government, Other'
WHERE OwnerTypes = 'Other, Local Government';

UPDATE staging
SET OwnerTypes = 'Non-Profit, Other'
WHERE OwnerTypes = 'Other, Non-Profit Org.';

UPDATE staging
SET OwnerTypes = 'Profit Org., Other'
WHERE OwnerTypes = 'Other, Profit Org.';

UPDATE staging
SET OwnerTypes = 'State Government, Other'
WHERE OwnerTypes = 'Other, State Government';

-- Update Individual, Other
UPDATE staging
SET OwnerTypes = 'Individual, Profit Org.'
WHERE OwnerTypes = 'Other, Individual, Profit Org.';

-- Update Private Sector: Private Developer
UPDATE staging
SET OwnerTypes = 'Private Sector: Private Developer'
WHERE OwnerTypes IN ('Provate', 'Private Developer', 'Private');

-- Update Profit Org.
UPDATE staging
SET OwnerTypes = 'Profit Org.'
WHERE OwnerTypes = 'Profit sector';

-- Update Public-Private Partnership (PPP)
UPDATE staging
SET OwnerTypes = 'Public-Private Partnership (PPP)'
WHERE OwnerTypes = 'Public Private Partnership';

-- Update Public Sector: College or University
UPDATE staging
SET OwnerTypes = 'Public Sector: College or University'
WHERE OwnerTypes LIKE 'Public Sector: College or Universit%';

-- Update Educational: University, Private
UPDATE staging
SET OwnerTypes = 'Educational: University, Private'
WHERE OwnerTypes = 'University';

COMMIT;

------------------------------------------------------------------------------

BEGIN TRANSACTION;

-- Convert GrossFloorArea from Sq ft to Sq m
UPDATE staging
SET 
    GrossFloorArea = CASE 
                        WHEN UnitOfMeasurement = 'Sq ft' AND GrossFloorArea IS NOT NULL 
                        THEN GrossFloorArea * 0.092903 
                        ELSE GrossFloorArea 
                     END,
    TotalPropArea = CASE 
                        WHEN UnitOfMeasurement = 'Sq ft' AND TotalPropArea IS NOT NULL 
                        THEN TotalPropArea * 0.092903 
                        ELSE TotalPropArea 
                    END,
    UnitOfMeasurement = CASE 
                            WHEN UnitOfMeasurement = 'Sq ft' 
                            THEN 'Sq m' 
                            ELSE UnitOfMeasurement 
                        END
WHERE UnitOfMeasurement = 'Sq ft' 
AND (GrossFloorArea IS NOT NULL OR TotalPropArea IS NOT NULL);

-- Verify the conversion for GrossFloorArea
SELECT GrossFloorArea, UnitOfMeasurement
FROM staging
WHERE UnitOfMeasurement = 'Sq m';

-- Verify the conversion for TotalPropArea
SELECT TotalPropArea, UnitOfMeasurement
FROM staging
WHERE UnitOfMeasurement = 'Sq m';

-- Round GrossFloorArea to one decimal place
UPDATE staging
SET GrossFloorArea = ROUND(GrossFloorArea, 1)
WHERE UnitOfMeasurement = 'Sq m';

-- Round TotalPropArea to one decimal place
UPDATE staging
SET TotalPropArea = ROUND(TotalPropArea, 1)
WHERE UnitOfMeasurement = 'Sq m';

ALTER TABLE staging ADD IsAreaInconsistent BIT;

-- Initialize all rows as consistent
UPDATE staging
SET IsAreaInconsistent = 0;


-- Flag rows with GrossFloorArea outside of the defined threshold
UPDATE staging
SET IsAreaInconsistent = 1
WHERE GrossFloorArea < 23;


-- Flag rows with TotalPropArea outside of the defined thresholds
UPDATE staging
SET IsAreaInconsistent = 1
WHERE TotalPropArea < 50
   OR TotalPropArea > 10000000;


-- Verify the consistency flagging
SELECT *
FROM staging
WHERE IsAreaInconsistent = 1;  -- View all inconsistent rows

SELECT *
FROM staging
WHERE IsAreaInconsistent = 0;  -- View all consistent rows

COMMIT;

select * from staging

UPDATE staging
SET ProjectTypes = 'Single-Family Home'
WHERE ProjectTypes = '0';

UPDATE staging
SET ProjectName = 'Recreation'
WHERE ProjectName = 'SEVINA PARK'

--2.1 Data Migration

-- Populate CertificationTimeline with the difference in days between RegistrationDate and CertDate
ALTER TABLE staging ADD CertificationTimeline INT;

UPDATE staging
SET CertificationTimeline = DATEDIFF(day, RegistrationDate, CertDate)
WHERE RegistrationDate IS NOT NULL AND CertDate IS NOT NULL;


BEGIN TRANSACTION;

-- Insert distinct OwnerTypes and OwnerOrganization into Owners table
INSERT INTO Owners (OwnerTypes, OwnerOrganization)
SELECT DISTINCT OwnerTypes, OwnerOrganization
FROM staging
WHERE OwnerTypes IS NOT NULL AND OwnerOrganization IS NOT NULL;

-- Insert data into Projects table using LEFT JOIN to include NULL Owner types
INSERT INTO Projects (ID, ProjectName, City, State, ZipCode, Country, GrossFloorArea, TotalPropArea, UnitOfMeasurement, ProjectTypes, OwnerID, IsAreaInconsistent)
SELECT 
    s.ID,
    s.ProjectName,
    s.City,
    s.State,
    s.ZipCode,
    s.Country,
    s.GrossFloorArea,
    s.TotalPropArea,
    s.UnitOfMeasurement,
    s.ProjectTypes,
    o.OwnerID,  -- Will be NULL if there is no match
    s.IsAreaInconsistent
FROM staging s
LEFT JOIN Owners o ON s.OwnerTypes = o.OwnerTypes AND s.OwnerOrganization = o.OwnerOrganization;


-- Insert data into Certifications table
INSERT INTO Certifications (ProjectID, LeedSystemVersionDisplayName, LeedVersion, PointsAchieved, CertLevel, RegistrationDate, CertDate, CertificationTimeline, IsCertified, IsDateSwapped)
SELECT 
    p.ProjectID,  -- Joining with Projects table to get ProjectID
    s.LeedSystemVersionDisplayName,
    s.LeedVersion,
    s.PointsAchieved,
    s.CertLevel,
    s.RegistrationDate,
    s.CertDate,
    s.CertificationTimeline,
    s.IsCertified,
    s.IsDateSwapped
FROM staging s
INNER JOIN Projects p ON s.ID = p.ID;

-- Insert data into Confidentiality table
INSERT INTO Confidentiality (ProjectID, IsConfidential)
SELECT 
    p.ProjectID,  -- Joining with Projects table to get ProjectID
    s.IsConfidential
FROM staging s
INNER JOIN Projects p ON s.ID = p.ID
WHERE s.IsConfidential IS NOT NULL;

SELECT *
FROM Projects
WHERE OwnerID IS NULL;

select * from Projects

COMMIT;

--save the staging table manually before archiving
--bcp "SELECT * FROM LEED_Project.dbo.staging" queryout "C:/Users/MONSTER/Desktop/LEED project/staging_table.csv" -c -t "|" -T -S DESKTOP-LLNSOOO

SELECT *
INTO archived_staging
FROM staging;

DROP TABLE staging; 

------------------------------------------------------------------------------------
--3.Analysis Tables

--Version and Certification Distributions Over Time 
CREATE VIEW vw_LeedVersionDistribution AS
WITH CertificationYears AS (
    SELECT 
        CASE 
            WHEN CertDate IS NOT NULL THEN YEAR(CertDate)
            WHEN CertDate IS NULL AND RegistrationDate IS NOT NULL THEN YEAR(RegistrationDate)
        END AS Year, 
        LeedVersion,
        IsCertified,
        COUNT(*) AS CertificationCount
    FROM Certifications c
    JOIN Projects p ON c.ProjectID = p.ProjectID
    WHERE p.Country = 'US'
    GROUP BY 
        CASE 
            WHEN CertDate IS NOT NULL THEN YEAR(CertDate)
            WHEN CertDate IS NULL AND RegistrationDate IS NOT NULL THEN YEAR(RegistrationDate)
        END,
        LeedVersion,
        IsCertified
    HAVING 
        CASE 
            WHEN CertDate IS NOT NULL THEN YEAR(CertDate)
            WHEN CertDate IS NULL AND RegistrationDate IS NOT NULL THEN YEAR(RegistrationDate)
        END IS NOT NULL
)
SELECT 
    Year,
    LeedVersion,
    SUM(CertificationCount) AS CertificationCount,
    FORMAT(ROUND(SUM(CertificationCount) * 100.0 / SUM(SUM(CertificationCount)) OVER (PARTITION BY Year), 2), 'N2') AS PercentageDistribution,
    SUM(CASE WHEN IsCertified = 1 THEN CertificationCount ELSE 0 END) AS CertifiedCount,
    SUM(CASE WHEN IsCertified = 0 THEN CertificationCount ELSE 0 END) AS NotCertifiedCount,
    FORMAT(ROUND(SUM(CASE WHEN IsCertified = 1 THEN CertificationCount ELSE 0 END) * 100.0 / SUM(CertificationCount), 2), 'N2') AS CertifiedPercentage,
    FORMAT(ROUND(SUM(CASE WHEN IsCertified = 0 THEN CertificationCount ELSE 0 END) * 100.0 / SUM(CertificationCount), 2), 'N2') AS NotCertifiedPercentage
FROM CertificationYears
GROUP BY Year, LeedVersion;

select * from vw_LeedVersionDistribution

/* NON NORMALIZED --Average Points Achieved by LEED Version and LEED Systems DECREPIT
CREATE VIEW vw_AvgPointsByLeedVersion AS
WITH ValidCertifications AS (
    SELECT 
        LeedVersion,
        LeedSystemVersionDisplayName,
        PointsAchieved
    FROM Certifications c
    JOIN Projects p ON c.ProjectID = p.ProjectID
    WHERE p.Country = 'US'
      AND PointsAchieved IS NOT NULL
)
SELECT 
    LeedVersion,
    LeedSystemVersionDisplayName,
    FORMAT(ROUND(AVG(PointsAchieved), 2), 'N2') AS AvgPoints
FROM ValidCertifications
GROUP BY 
    GROUPING SETS (
        (LeedVersion, LeedSystemVersionDisplayName), 
        (LeedVersion)
    );

*/

--Average Points Achieved by LEED Version and LEED Systems(v1/v2 NORMALIZED)
CREATE VIEW vw_AvgPointsByLeedVersion AS
WITH ValidCertifications AS (
    SELECT 
        LeedVersion,
        LeedSystemVersionDisplayName,
        CASE 
            -- Normalize LEED v2 points
            WHEN LeedVersion IN ('v2','v1') THEN 
                CASE 
                    WHEN PointsAchieved BETWEEN 26 AND 32 THEN (PointsAchieved - 26) / (32 - 26) * (49 - 40) + 40
                    WHEN PointsAchieved BETWEEN 33 AND 38 THEN (PointsAchieved - 33) / (38 - 33) * (59 - 50) + 50
                    WHEN PointsAchieved BETWEEN 39 AND 51 THEN (PointsAchieved - 39) / (51 - 39) * (79 - 60) + 60
                    WHEN PointsAchieved BETWEEN 52 AND 69 THEN (PointsAchieved - 52) / (69 - 52) * (110 - 80) + 80
                    ELSE PointsAchieved  -- If points don't fall into expected ranges, leave them as-is
                END
            ELSE 
                PointsAchieved  -- For LEED v3 and v4, leave points as they are
        END AS NormalizedPoints
    FROM Certifications c
    JOIN Projects p ON c.ProjectID = p.ProjectID
    WHERE p.Country = 'US'
      AND PointsAchieved IS NOT NULL
)
SELECT 
    LeedVersion,
    LeedSystemVersionDisplayName,
    FORMAT(ROUND(AVG(NormalizedPoints), 2), 'N2') AS AvgPoints
FROM ValidCertifications
GROUP BY 
    GROUPING SETS (
        (LeedVersion, LeedSystemVersionDisplayName), 
        (LeedVersion)
    );


select * from vw_AvgPointsByLeedVersion order by LeedVersion


--Certification Time Difference between registration and certification by LEED Version and LEED Systems
CREATE VIEW vw_CertificationTimelineByLeedVersion AS
WITH ValidCertifications AS (
    SELECT 
        LeedVersion,
        LeedSystemVersionDisplayName,
        CertificationTimeline
    FROM Certifications c
    JOIN Projects p ON c.ProjectID = p.ProjectID
    WHERE p.Country = 'US'
      AND CertificationTimeline IS NOT NULL
)
SELECT 
    LeedVersion,
    LeedSystemVersionDisplayName,
    AVG(CertificationTimeline) AS AvgCertificationTimeline
FROM ValidCertifications
GROUP BY 
    GROUPING SETS (
        (LeedVersion, LeedSystemVersionDisplayName), 
        (LeedVersion)
    );


select * from vw_CertificationTimelineByLeedVersion order by LeedVersion

--Certification Level Distribution by LEED Version and LEED Systems
CREATE VIEW vw_CertificationLevelDistribution AS
WITH ValidCertifications AS (
    SELECT 
        LeedVersion,
        LeedSystemVersionDisplayName,
        CertLevel,
        COUNT(*) AS CertificationCount
    FROM Certifications c
    JOIN Projects p ON c.ProjectID = p.ProjectID
    WHERE p.Country = 'US'
      AND CertLevel IS NOT NULL
      AND CertLevel <> 0
    GROUP BY 
        LeedVersion, 
        LeedSystemVersionDisplayName, 
        CertLevel
)
SELECT 
    LeedVersion,
    LeedSystemVersionDisplayName,
    CertLevel,
    CertificationCount,
    FORMAT(ROUND(CertificationCount * 100.0 / SUM(CertificationCount) OVER (PARTITION BY LeedVersion, CertLevel), 2), 'N2') AS PercentageWithinCertLevel,
    FORMAT(ROUND(SUM(CertificationCount) OVER (PARTITION BY LeedVersion, CertLevel) * 100.0 / SUM(SUM(CertificationCount)) OVER (PARTITION BY LeedVersion), 2), 'N2') AS PercentageWithinLeedVersion
FROM ValidCertifications
GROUP BY 
    LeedVersion,
    LeedSystemVersionDisplayName,
    CertLevel,
    CertificationCount;

select * from vw_CertificationLevelDistribution order by LeedVersion, CertLevel

--Yearly Growth Rate of Certifications by LEED Version
CREATE VIEW vw_YearlyGrowthRateByLeedVersion AS
WITH YearlyCertifications AS (
    SELECT 
        YEAR(CertDate) AS Year, 
        LeedVersion, 
        COUNT(*) AS CertificationCount
    FROM Certifications c
    JOIN Projects p ON c.ProjectID = p.ProjectID
    WHERE p.Country = 'US'
    GROUP BY YEAR(CertDate), LeedVersion
),
LaggedCertifications AS (
    SELECT
        Year,
        LeedVersion,
        CertificationCount,
        LAG(CertificationCount, 1) OVER (PARTITION BY LeedVersion ORDER BY Year) AS PreviousYearCount
    FROM YearlyCertifications
)
SELECT 
    Year,
    LeedVersion,
    CertificationCount,
    PreviousYearCount,
    ROUND((CertificationCount - PreviousYearCount) * 100.0 / PreviousYearCount, 2) AS GrowthRate
FROM LaggedCertifications
WHERE PreviousYearCount IS NOT NULL;

select * from vw_YearlyGrowthRateByLeedVersion order by Year,LeedVersion

--Cumulative Adoption of LEED Versions Over Time
CREATE VIEW vw_CumulativeAdoptionByLeedVersion AS
WITH YearlyCertifications AS (
    SELECT 
        YEAR(CertDate) AS Year, 
        LeedVersion, 
        COUNT(*) AS CertificationCount
    FROM Certifications c
    JOIN Projects p ON c.ProjectID = p.ProjectID
    WHERE p.Country = 'US'
    GROUP BY YEAR(CertDate), LeedVersion
)
SELECT 
    Year, 
    LeedVersion, 
    CertificationCount,
    SUM(CertificationCount) OVER (PARTITION BY LeedVersion ORDER BY Year) AS CumulativeCertifications
FROM YearlyCertifications;

select * from vw_CumulativeAdoptionByLeedVersion order by Year, LeedVersion

--2. Analysis Item: Typology and Project Size Influence on LEED Certification

--Prior to the keyword extraction, some rows containing ":" as delimiter while most using only commas, so altering the occurances of colon to comma is done first
CREATE VIEW PreprocessedProjectTypes AS
SELECT 
    ProjectID,
    REPLACE(ProjectTypes, ':', ',') AS ProcessedProjectType,
    GrossFloorArea,
    TotalPropArea,
    Country,
	IsAreaInconsistent
FROM 
    Projects;


--As the rows of ProjectTypes containing multiple keywords, extracting each keyword separated by commas and counting them one by one has been decided.
CREATE VIEW ProjectTypeKeywordCounts AS
WITH SplitKeywords AS (
    SELECT 
        ProjectID,
        TRIM(value) AS Keyword
    FROM 
        PreprocessedProjectTypes
    CROSS APPLY 
        STRING_SPLIT(ProcessedProjectType, ',')
    WHERE 
        Country = 'US'
)
SELECT 
    Keyword,
    COUNT(*) AS KeywordCount
FROM 
    SplitKeywords
GROUP BY 
    Keyword
ORDER BY 
    KeywordCount DESC
OFFSET 0 ROWS FETCH NEXT 100 ROWS ONLY;

drop view ProjectTypeKeywordCounts

select * from ProjectTypeKeywordCounts


/*--Project type analysis via certlevel and points achieved
CREATE VIEW ProjectTypeTopKeywordsCertLevel AS
WITH SplitKeywords AS (
    SELECT 
        ProjectID,
        TRIM(value) AS Keyword
    FROM 
        PreprocessedProjectTypes
    CROSS APPLY 
        STRING_SPLIT(ProcessedProjectType, ',')
    WHERE 
        Country = 'US'
),
BinnedKeywords AS (
    SELECT 
        ProjectID,
        CASE 
            WHEN Keyword IN ('Single-Family Home', 'detached single-family', 'Single-Family (Detached)', 'Single family detached', 'Single-Family Home (Detached)', 'Single-Family (Attached)', 'Attached single-family', 'Single family attached', 'Single-Family Home (Attached)') THEN 'Single-Family Homes'
            WHEN Keyword IN ('Multi-Unit Residence', 'Multi-Family Residential', 'Low-rise multi-family', 'Multi-Family Low-Rise (1-3 stories)', 'Multi-Family Mid-Rise', 'Mid-rise multi-family', 'Multifamily Lowrise', 'Multifamily Midrise', 'Multi-Family Low-Rise Building (1-3 stories)') THEN 'Multi-Family Residential'
            WHEN Keyword IN ('Dormitory', 'Nursing Home/ Assisted Living', 'Condominium', 'Affordable Housing', 'Special Needs', 'Clinic/Other Outpatient') THEN 'Specialized Residential'
            WHEN Keyword IN ('Higher Education', 'College/University', 'Campus (corp/school)') THEN 'Higher Education'
            WHEN Keyword IN ('K-12 Education', 'K-12 Elementary/Middle School', 'K-12 High School', 'Elementary/Middle School', 'High School', 'Other classroom education', 'K-12') THEN 'K-12 Education'
            WHEN Keyword IN ('Preschool/Daycare', 'Daycare') THEN 'Preschool/Daycare'
            WHEN Keyword IN ('Government', 'Military Base') THEN 'Federal Government'
            WHEN Keyword IN ('Public Order and Safety', 'Public Order/Safety', 'Fire/Police Station', 'Other Public Order', 'Community Dev.', 'Interpretive Center') THEN 'Local Government/Public Order'
            WHEN Keyword IN ('Public Assembly', 'Assembly', 'Other Assembly', 'Stadium/Arena') THEN 'Public Assembly'
            WHEN Keyword IN ('Airport', 'Vehicle Storage/Maintenance', 'Vehicle Dealership', 'Vehicle Service/Repair', 'Transportation') THEN 'Transportation'
            WHEN Keyword IN ('Health Care', 'Healthcare', 'Clinic/Other Outpatient', 'Medical (Non-Diagnostic)', 'Inpatient', 'Outpatient Office (Diagnostic)') THEN 'General Healthcare'
            WHEN Keyword IN ('Laboratory', 'Nursing Home/ Assisted Living') THEN 'Specialized Healthcare'
            WHEN Keyword IN ('Office', 'Commercial Office', 'Administrative/Professional', 'Other Office', 'Outpatient Office (Diagnostic)') THEN 'General Office'
            WHEN Keyword IN ('Bank Branch', 'Financial', 'Financial & Comm.') THEN 'Financial'
            WHEN Keyword IN ('Retail', 'Other Retail', 'Restaurant', 'Restaurant/Cafeteria', 'Fast Food', 'Convenience Store', 'Grocery Store/Food Market', 'Open Shopping Center') THEN 'Retail'
            WHEN Keyword IN ('Lodging', 'Hotel/Motel/Resort', 'Hotel/Resort', 'Other lodging', 'Full Service', 'Limited Service') THEN 'Lodging'
            WHEN Keyword IN ('Industrial', 'Industrial Manufacturing', 'Warehouse', 'Warehouse and Distribution Center', 'Nonrefrigerated Distribution/Shipping', 'Data Center') THEN 'Industrial'
            WHEN Keyword IN ('Service', 'Other Service', 'Entertainment', 'Social/Meeting', 'Recreation', 'Animal Care') THEN 'Service'
            WHEN Keyword IN ('Confidential') THEN 'Confidential'
            WHEN Keyword IN ('General') THEN 'General'
            WHEN Keyword IN ('Core Learning Space') THEN 'Core Learning Space'
            WHEN Keyword IN ('Other') THEN 'Other'
            WHEN Keyword IN ('Park (eg. greenway)') THEN 'Park'
            ELSE 'Other'
        END AS BinnedKeyword
    FROM 
        SplitKeywords
)
SELECT 
    bk.BinnedKeyword,
    cl.CertLevel,
    COUNT(*) AS ProjectCount,
    FORMAT(ROUND(AVG(cl.PointsAchieved),2), 'N2') AS AvgPointsAchieved
FROM 
    BinnedKeywords bk 
INNER JOIN 
    Certifications cl ON bk.ProjectID = cl.ProjectID
WHERE 
    cl.CertLevel NOT IN (0, 2)
GROUP BY 
    bk.BinnedKeyword, cl.CertLevel


select * from ProjectTypeTopKeywordsCertLevel order by CertLevel,ProjectCount
*/


--Project type analysis via certlevel and points achieved
CREATE VIEW ProjectTypeTopKeywordsCertLevel_norm AS
WITH SplitKeywords AS (
    SELECT 
        ProjectID,
        TRIM(value) AS Keyword
    FROM 
        PreprocessedProjectTypes
    CROSS APPLY 
        STRING_SPLIT(ProcessedProjectType, ',')
    WHERE 
        Country = 'US'
),
BinnedKeywords AS (
    SELECT 
        p.ProjectID,
        CASE 
            -- Normalize LEED v2 points
            WHEN c.LeedVersion IN ('v2','v1') THEN 
                CASE 
                    WHEN c.PointsAchieved BETWEEN 26 AND 32 THEN (c.PointsAchieved - 26) / (32 - 26) * (49 - 40) + 40
                    WHEN c.PointsAchieved BETWEEN 33 AND 38 THEN (c.PointsAchieved - 33) / (38 - 33) * (59 - 50) + 50
                    WHEN c.PointsAchieved BETWEEN 39 AND 51 THEN (c.PointsAchieved - 39) / (51 - 39) * (79 - 60) + 60
                    WHEN c.PointsAchieved BETWEEN 52 AND 69 THEN (c.PointsAchieved - 52) / (69 - 52) * (110 - 80) + 80
                    ELSE c.PointsAchieved  -- If points don't fall into expected ranges, leave them as-is
                END
            ELSE 
                c.PointsAchieved  -- For LEED v3 and v4, leave points as they are
        END AS NormalizedPoints,
        CASE 
            WHEN Keyword IN ('Single-Family Home', 'detached single-family', 'Single-Family (Detached)', 'Single family detached', 'Single-Family Home (Detached)', 'Single-Family (Attached)', 'Attached single-family', 'Single family attached', 'Single-Family Home (Attached)') THEN 'Single-Family Homes'
            WHEN Keyword IN ('Multi-Unit Residence', 'Multi-Family Residential', 'Low-rise multi-family', 'Multi-Family Low-Rise (1-3 stories)', 'Multi-Family Mid-Rise', 'Mid-rise multi-family', 'Multifamily Lowrise', 'Multifamily Midrise', 'Multi-Family Low-Rise Building (1-3 stories)') THEN 'Multi-Family Residential'
            WHEN Keyword IN ('Dormitory', 'Nursing Home/ Assisted Living', 'Condominium', 'Affordable Housing', 'Special Needs', 'Clinic/Other Outpatient') THEN 'Specialized Residential'
            WHEN Keyword IN ('Higher Education', 'College/University', 'Campus (corp/school)') THEN 'Higher Education'
            WHEN Keyword IN ('K-12 Education', 'K-12 Elementary/Middle School', 'K-12 High School', 'Elementary/Middle School', 'High School', 'Other classroom education', 'K-12') THEN 'K-12 Education'
            WHEN Keyword IN ('Preschool/Daycare', 'Daycare') THEN 'Preschool/Daycare'
            WHEN Keyword IN ('Government', 'Military Base') THEN 'Federal Government'
            WHEN Keyword IN ('Public Order and Safety', 'Public Order/Safety', 'Fire/Police Station', 'Other Public Order', 'Community Dev.', 'Interpretive Center') THEN 'Local Government/Public Order'
            WHEN Keyword IN ('Public Assembly', 'Assembly', 'Other Assembly', 'Stadium/Arena') THEN 'Public Assembly'
            WHEN Keyword IN ('Airport', 'Vehicle Storage/Maintenance', 'Vehicle Dealership', 'Vehicle Service/Repair', 'Transportation') THEN 'Transportation'
            WHEN Keyword IN ('Health Care', 'Healthcare', 'Clinic/Other Outpatient', 'Medical (Non-Diagnostic)', 'Inpatient', 'Outpatient Office (Diagnostic)') THEN 'General Healthcare'
            WHEN Keyword IN ('Laboratory', 'Nursing Home/ Assisted Living') THEN 'Specialized Healthcare'
            WHEN Keyword IN ('Office', 'Commercial Office', 'Administrative/Professional', 'Other Office', 'Outpatient Office (Diagnostic)') THEN 'General Office'
            WHEN Keyword IN ('Bank Branch', 'Financial', 'Financial & Comm.') THEN 'Financial'
            WHEN Keyword IN ('Retail', 'Other Retail', 'Restaurant', 'Restaurant/Cafeteria', 'Fast Food', 'Convenience Store', 'Grocery Store/Food Market', 'Open Shopping Center') THEN 'Retail'
            WHEN Keyword IN ('Lodging', 'Hotel/Motel/Resort', 'Hotel/Resort', 'Other lodging', 'Full Service', 'Limited Service') THEN 'Lodging'
            WHEN Keyword IN ('Industrial', 'Industrial Manufacturing', 'Warehouse', 'Warehouse and Distribution Center', 'Nonrefrigerated Distribution/Shipping', 'Data Center') THEN 'Industrial'
            WHEN Keyword IN ('Service', 'Other Service', 'Entertainment', 'Social/Meeting', 'Recreation', 'Animal Care') THEN 'Service'
            WHEN Keyword IN ('Confidential') THEN 'Confidential'
            WHEN Keyword IN ('General') THEN 'General'
            WHEN Keyword IN ('Core Learning Space') THEN 'Core Learning Space'
            WHEN Keyword IN ('Other') THEN 'Other'
            WHEN Keyword IN ('Park (eg. greenway)') THEN 'Park'
            ELSE 'Other'
        END AS BinnedKeyword
    FROM 
        SplitKeywords sk
    JOIN 
        Projects p ON sk.ProjectID = p.ProjectID
    JOIN 
        Certifications c ON p.ProjectID = c.ProjectID
)
SELECT 
    bk.BinnedKeyword,
    cl.CertLevel,
    COUNT(*) AS ProjectCount,
    ROUND(AVG(bk.NormalizedPoints), 2) AS AvgPointsAchieved
FROM 
    BinnedKeywords bk
JOIN 
    Certifications cl ON bk.ProjectID = cl.ProjectID
WHERE 
    cl.CertLevel NOT IN (0, 2)
GROUP BY 
    bk.BinnedKeyword, cl.CertLevel;


select * from ProjectTypeTopKeywordsCertLevel_norm order by CertLevel,ProjectCount


CREATE VIEW GrossFloorAreaCertLevelImpact AS
WITH SplitKeywords AS (
    SELECT 
        pt.ProjectID,
        pt.GrossFloorArea,
        pt.TotalPropArea,
        cl.CertLevel,
        cl.PointsAchieved,
        cl.LeedVersion,  -- Ensure LeedVersion is selected here
        TRIM(value) AS Keyword
    FROM 
        PreprocessedProjectTypes pt
    CROSS APPLY 
        STRING_SPLIT(pt.ProcessedProjectType, ',') AS sk
    JOIN 
        Certifications cl ON pt.ProjectID = cl.ProjectID
    WHERE 
        pt.Country = 'US'
        AND pt.IsAreaInconsistent = 0
        AND pt.GrossFloorArea IS NOT NULL
        AND (pt.TotalPropArea IS NULL OR pt.TotalPropArea > 0)
),
BinnedData AS (
    SELECT 
        sk.ProjectID,
        sk.GrossFloorArea,
        sk.TotalPropArea,
        sk.CertLevel,
        sk.LeedVersion,  -- Carry LeedVersion through to this CTE
        CASE 
            -- Normalize LEED v2 points
            WHEN sk.LeedVersion IN ('v2', 'v1') THEN 
                CASE 
                    WHEN sk.PointsAchieved BETWEEN 26 AND 32 THEN (sk.PointsAchieved - 26) / (32 - 26) * (49 - 40) + 40
                    WHEN sk.PointsAchieved BETWEEN 33 AND 38 THEN (sk.PointsAchieved - 33) / (38 - 33) * (59 - 50) + 50
                    WHEN sk.PointsAchieved BETWEEN 39 AND 51 THEN (sk.PointsAchieved - 39) / (51 - 39) * (79 - 60) + 60
                    WHEN sk.PointsAchieved BETWEEN 52 AND 69 THEN (sk.PointsAchieved - 52) / (69 - 52) * (110 - 80) + 80
                    ELSE sk.PointsAchieved  -- If points don't fall into expected ranges, leave them as-is
                END
            ELSE 
                sk.PointsAchieved  -- For LEED v3 and v4, leave points as they are
        END AS NormalizedPointsAchieved,
        CASE 
            WHEN sk.Keyword IN ('Single-Family Home', 'detached single-family', 'Single-Family (Detached)', 'Single family detached', 'Single-Family Home (Detached)', 'Single-Family (Attached)', 'Attached single-family', 'Single family attached', 'Single-Family Home (Attached)') THEN 'Single-Family Homes'
            WHEN sk.Keyword IN ('Multi-Unit Residence', 'Multi-Family Residential', 'Low-rise multi-family', 'Multi-Family Low-Rise (1-3 stories)', 'Multi-Family Mid-Rise', 'Mid-rise multi-family', 'Multifamily Lowrise', 'Multifamily Midrise', 'Multi-Family Low-Rise Building (1-3 stories)') THEN 'Multi-Family Residential'
            WHEN sk.Keyword IN ('Dormitory', 'Nursing Home/ Assisted Living', 'Condominium', 'Affordable Housing', 'Special Needs', 'Clinic/Other Outpatient') THEN 'Specialized Residential'
            WHEN sk.Keyword IN ('Higher Education', 'College/University', 'Campus (corp/school)') THEN 'Higher Education'
            WHEN sk.Keyword IN ('K-12 Education', 'K-12 Elementary/Middle School', 'K-12 High School', 'Elementary/Middle School', 'High School', 'Other classroom education', 'K-12') THEN 'K-12 Education'
            WHEN sk.Keyword IN ('Preschool/Daycare', 'Daycare') THEN 'Preschool/Daycare'
            WHEN sk.Keyword IN ('Government', 'Military Base') THEN 'Federal Government'
            WHEN sk.Keyword IN ('Public Order and Safety', 'Public Order/Safety', 'Fire/Police Station', 'Other Public Order', 'Community Dev.', 'Interpretive Center') THEN 'Local Government/Public Order'
            WHEN sk.Keyword IN ('Public Assembly', 'Assembly', 'Other Assembly', 'Stadium/Arena') THEN 'Public Assembly'
            WHEN sk.Keyword IN ('Airport', 'Vehicle Storage/Maintenance', 'Vehicle Dealership', 'Vehicle Service/Repair', 'Transportation') THEN 'Transportation'
            WHEN sk.Keyword IN ('Health Care', 'Healthcare', 'Clinic/Other Outpatient', 'Medical (Non-Diagnostic)', 'Inpatient', 'Outpatient Office (Diagnostic)') THEN 'General Healthcare'
            WHEN sk.Keyword IN ('Laboratory', 'Nursing Home/ Assisted Living') THEN 'Specialized Healthcare'
            WHEN sk.Keyword IN ('Office', 'Commercial Office', 'Administrative/Professional', 'Other Office', 'Outpatient Office (Diagnostic)') THEN 'General Office'
            WHEN sk.Keyword IN ('Bank Branch', 'Financial', 'Financial & Comm.') THEN 'Financial'
            WHEN sk.Keyword IN ('Retail', 'Other Retail', 'Restaurant', 'Restaurant/Cafeteria', 'Fast Food', 'Convenience Store', 'Grocery Store/Food Market', 'Open Shopping Center') THEN 'Retail'
            WHEN sk.Keyword IN ('Lodging', 'Hotel/Motel/Resort', 'Hotel/Resort', 'Other lodging', 'Full Service', 'Limited Service') THEN 'Lodging'
            WHEN sk.Keyword IN ('Industrial', 'Industrial Manufacturing', 'Warehouse', 'Warehouse and Distribution Center', 'Nonrefrigerated Distribution/Shipping', 'Data Center') THEN 'Industrial'
            WHEN sk.Keyword IN ('Service', 'Other Service', 'Entertainment', 'Social/Meeting', 'Recreation', 'Animal Care') THEN 'Service'
            WHEN sk.Keyword = 'Confidential' THEN 'Confidential'
            WHEN sk.Keyword = 'General' THEN 'General'
            WHEN sk.Keyword = 'Core Learning Space' THEN 'Core Learning Space'
            WHEN sk.Keyword = 'Other' THEN 'Other'
            WHEN sk.Keyword = 'Park (eg. greenway)' THEN 'Park'
            ELSE 'Other'
        END AS BinnedProjectType,
        CASE
            WHEN sk.GrossFloorArea BETWEEN 50 AND 500 THEN 'Micro'
            WHEN sk.GrossFloorArea BETWEEN 501 AND 5000 THEN 'Small'
            WHEN sk.GrossFloorArea BETWEEN 5001 AND 20000 THEN 'Moderate'
            WHEN sk.GrossFloorArea BETWEEN 20001 AND 100000 THEN 'Large'
            WHEN sk.GrossFloorArea BETWEEN 100001 AND 500000 THEN 'Very Large'
            WHEN sk.GrossFloorArea > 500000 THEN 'Mega'
            ELSE 'Unknown'
        END AS SizeCategory,
        CASE 
            WHEN sk.TotalPropArea IS NOT NULL AND sk.TotalPropArea > 0 THEN sk.GrossFloorArea / sk.TotalPropArea
            ELSE NULL
        END AS GFATPARatio,
        CASE 
            WHEN sk.TotalPropArea IS NOT NULL AND sk.TotalPropArea > 0 THEN 
                CASE 
                    WHEN (sk.GrossFloorArea / sk.TotalPropArea) BETWEEN 0 AND 0.1 THEN 'Low Density'
                    WHEN (sk.GrossFloorArea / sk.TotalPropArea) BETWEEN 0.1 AND 0.25 THEN 'Medium-Low Density'
                    WHEN (sk.GrossFloorArea / sk.TotalPropArea) BETWEEN 0.25 AND 0.5 THEN 'Medium Density'
                    WHEN (sk.GrossFloorArea / sk.TotalPropArea) BETWEEN 0.5 AND 1.0 THEN 'Medium-High Density'
                    WHEN (sk.GrossFloorArea / sk.TotalPropArea) BETWEEN 1.0 AND 2.0 THEN 'High Density'
                    WHEN (sk.GrossFloorArea / sk.TotalPropArea) > 2.0 THEN 'Ultra-High Density'
                    ELSE 'Unknown Density'
                END
            ELSE 'No TPA'
        END AS DensityCategory
    FROM 
        SplitKeywords sk
)
SELECT 
    BinnedData.ProjectID,
    BinnedData.BinnedProjectType,
    BinnedData.GrossFloorArea,
    BinnedData.TotalPropArea,
    BinnedData.CertLevel,
    ROUND(BinnedData.NormalizedPointsAchieved,2) AS NormalizedPointsAchieved,  -- Use normalized points
    BinnedData.SizeCategory,
    ROUND(BinnedData.GFATPARatio, 2) AS GFATPARatio,
    BinnedData.DensityCategory
FROM 
    BinnedData;


select * from GrossFloorAreaCertLevelImpact


--State level distribution of certifications
CREATE VIEW StateLevelLEEDDistribution AS
WITH NormalizedCertifications AS (
    SELECT 
        pt.State,
        CASE 
            -- Normalize LEED v2 points
            WHEN cl.LeedVersion IN ('v2', 'v1') THEN 
                CASE 
                    WHEN cl.PointsAchieved BETWEEN 26 AND 32 THEN (cl.PointsAchieved - 26) / (32 - 26) * (49 - 40) + 40
                    WHEN cl.PointsAchieved BETWEEN 33 AND 38 THEN (cl.PointsAchieved - 33) / (38 - 33) * (59 - 50) + 50
                    WHEN cl.PointsAchieved BETWEEN 39 AND 51 THEN (cl.PointsAchieved - 39) / (51 - 39) * (79 - 60) + 60
                    WHEN cl.PointsAchieved BETWEEN 52 AND 69 THEN (cl.PointsAchieved - 52) / (69 - 52) * (110 - 80) + 80
                    ELSE cl.PointsAchieved  -- If points don't fall into expected ranges, leave them as-is
                END
            ELSE 
                cl.PointsAchieved  -- For LEED v3 and v4, leave points as they are
        END AS NormalizedPointsAchieved,
        cl.CertLevel
    FROM 
        Projects pt
    JOIN 
        Certifications cl ON pt.ProjectID = cl.ProjectID
    WHERE 
        pt.Country = 'US' AND
		pt.State IS NOT NULL
)
SELECT 
    nc.State,
    COUNT(nc.CertLevel) AS TotalCertifications,
    ROUND(AVG(nc.NormalizedPointsAchieved), 2) AS AvgPointsAchieved,
    COUNT(CASE WHEN nc.CertLevel = '1' THEN 1 END) AS CertifiedCount,
    COUNT(CASE WHEN nc.CertLevel = '3' THEN 1 END) AS SilverCount,
    COUNT(CASE WHEN nc.CertLevel = '4' THEN 1 END) AS GoldCount,
    COUNT(CASE WHEN nc.CertLevel = '5' THEN 1 END) AS PlatinumCount
FROM 
    NormalizedCertifications nc
GROUP BY 
    nc.State;


select * from StateLevelLEEDDistribution ORDER BY TotalCertifications DESC;

CREATE TABLE ClimateZones (
    State VARCHAR(2),
    ClimateZone VARCHAR(10)
);

-- Insert the data into the ClimateZones table
INSERT INTO ClimateZones (State, ClimateZone) VALUES
('AL', '3A'),
('AK', '7'),
('AZ', '2B'),
('AR', '3A'),
('CA', '3B'),
('CA', '3C'),
('CA', '4C'),
('CO', '5B'),
('CT', '5A'),
('DE', '4A'),
('FL', '2A'),
('GA', '3A'),
('HI', '1A'),
('ID', '5B'),
('ID', '6B'),
('IL', '5A'),
('IN', '5A'),
('IA', '6A'),
('KS', '4A'),
('KS', '5A'),
('KY', '4A'),
('LA', '2A'),
('ME', '6A'),
('MD', '4A'),
('MA', '5A'),
('MI', '6A'),
('MN', '6A'),
('MN', '7'),
('MS', '3A'),
('MO', '4A'),
('MO', '5A'),
('MT', '6B'),
('MT', '7'),
('NE', '5A'),
('NV', '3B'),
('NV', '5B'),
('NH', '6A'),
('NJ', '4A'),
('NM', '4B'),
('NM', '5B'),
('NY', '4A'),
('NY', '5A'),
('NC', '3A'),
('NC', '4A'),
('ND', '6A'),
('ND', '7'),
('OH', '5A'),
('OK', '3A'),
('OK', '4A'),
('OR', '4C'),
('OR', '5B'),
('PA', '4A'),
('PA', '5A'),
('RI', '5A'),
('SC', '3A'),
('SD', '5B'),
('SD', '6A'),
('TN', '3A'),
('TN', '4A'),
('TX', '2A'),
('TX', '2B'),
('TX', '3A'),
('TX', '3B'),
('UT', '5B'),
('VT', '6A'),
('VA', '4A'),
('WA', '4C'),
('WA', '5B'),
('WV', '4A'),
('WI', '6A'),
('WY', '5B'),
('WY', '6B'),
('DC', '4A');
--Cert levels across climate zones, by ASHRAE classification of the states
CREATE VIEW ClimateZoneLEEDAnalysis AS
WITH StateClimateData AS (
    SELECT 
        pt.State,
        cz.ClimateZone,
        cl.CertLevel,
        CASE 
            -- Normalize LEED v2 points
            WHEN cl.LeedVersion IN ('v2', 'v1') THEN 
                CASE 
                    WHEN cl.PointsAchieved BETWEEN 26 AND 32 THEN (cl.PointsAchieved - 26) / (32 - 26) * (49 - 40) + 40
                    WHEN cl.PointsAchieved BETWEEN 33 AND 38 THEN (cl.PointsAchieved - 33) / (38 - 33) * (59 - 50) + 50
                    WHEN cl.PointsAchieved BETWEEN 39 AND 51 THEN (cl.PointsAchieved - 39) / (51 - 39) * (79 - 60) + 60
                    WHEN cl.PointsAchieved BETWEEN 52 AND 69 THEN (cl.PointsAchieved - 52) / (69 - 52) * (110 - 80) + 80
                    ELSE cl.PointsAchieved  -- If points don't fall into expected ranges, leave them as-is
                END
            ELSE 
                cl.PointsAchieved  -- For LEED v3 and v4, leave points as they are
        END AS NormalizedPointsAchieved
    FROM 
        Projects pt
    JOIN 
        Certifications cl ON pt.ProjectID = cl.ProjectID
    JOIN 
        ClimateZones cz ON pt.State = cz.State
    WHERE 
        pt.Country = 'US'
)
SELECT 
    scd.ClimateZone,
    COUNT(DISTINCT scd.State) AS NumberOfStates,
    COUNT(scd.CertLevel) AS TotalCertifications,
    ROUND(AVG(scd.NormalizedPointsAchieved), 2) AS AvgPointsAchieved,
    COUNT(CASE WHEN scd.CertLevel = '1' THEN 1 END) AS CertifiedCount,
    COUNT(CASE WHEN scd.CertLevel = '3' THEN 1 END) AS SilverCount,
    COUNT(CASE WHEN scd.CertLevel = '4' THEN 1 END) AS GoldCount,
    COUNT(CASE WHEN scd.CertLevel = '5' THEN 1 END) AS PlatinumCount
FROM 
    StateClimateData scd
GROUP BY 
    scd.ClimateZone;


select * from ClimateZoneLEEDAnalysis order by TotalCertifications DESC;


-- Certification performance across owner types
CREATE VIEW DetailedOwnerTypeImpactWithTotals AS
WITH NormalizedCertifications AS (
    SELECT 
        CASE 
            WHEN o.OwnerTypes LIKE '%Government%' THEN 'Government'
            WHEN o.OwnerTypes LIKE '%Non-Profit%' OR o.OwnerTypes LIKE '%Non%' OR o.OwnerTypes LIKE '%Community Development%' THEN 'Non-Profit'
            WHEN o.OwnerTypes LIKE '%Educational%' OR o.OwnerTypes LIKE '%School%' OR o.OwnerTypes LIKE '%University%' OR o.OwnerTypes LIKE '%College%' THEN 'Educational'
            WHEN o.OwnerTypes LIKE '%Profit%' OR o.OwnerTypes LIKE '%Private%' OR o.OwnerTypes LIKE '%Corporate%' OR o.OwnerTypes LIKE '%Investor%' OR o.OwnerTypes LIKE '%REIT%' THEN 'Private Sector'
            WHEN o.OwnerTypes LIKE '%Individual%' OR o.OwnerTypes LIKE '%Family%' THEN 'Individual/Family'
            WHEN o.OwnerTypes LIKE '%Residential%' OR o.OwnerTypes LIKE '%Landlord%' THEN 'Residential'
            WHEN o.OwnerTypes LIKE '%Religious%' THEN 'Religious'
            WHEN o.OwnerTypes LIKE '%Urban Development%' OR o.OwnerTypes LIKE '%Improvement District%' OR o.OwnerTypes LIKE '%Main Street%' THEN 'Urban Development'
            WHEN o.OwnerTypes LIKE '%Public-Private%' THEN 'Public-Private Partnership'
            WHEN o.OwnerTypes LIKE '%Confidential%' THEN 'Confidential'
            ELSE 'Other'
        END AS BinnedOwnerType,
        o.OwnerTypes AS SpecificOwnerType,
        cl.CertLevel,
        CASE 
            -- Normalize LEED v2 points
            WHEN cl.LeedVersion IN ('v2', 'v1') THEN 
                CASE 
                    WHEN cl.PointsAchieved BETWEEN 26 AND 32 THEN (cl.PointsAchieved - 26) / (32 - 26) * (49 - 40) + 40
                    WHEN cl.PointsAchieved BETWEEN 33 AND 38 THEN (cl.PointsAchieved - 33) / (38 - 33) * (59 - 50) + 50
                    WHEN cl.PointsAchieved BETWEEN 39 AND 51 THEN (cl.PointsAchieved - 39) / (51 - 39) * (79 - 60) + 60
                    WHEN cl.PointsAchieved BETWEEN 52 AND 69 THEN (cl.PointsAchieved - 52) / (69 - 52) * (110 - 80) + 80
                    ELSE cl.PointsAchieved  -- If points don't fall into expected ranges, leave them as-is
                END
            ELSE 
                cl.PointsAchieved  -- For LEED v3 and v4, leave points as they are
        END AS NormalizedPointsAchieved
    FROM 
        Projects p
    JOIN 
        Certifications cl ON p.ProjectID = cl.ProjectID
    JOIN 
        Owners o ON p.OwnerID = o.OwnerID
    WHERE 
        p.Country = 'US'
)
SELECT 
    COALESCE(nc.BinnedOwnerType, 'Total') AS BinnedOwnerType,
    CASE WHEN GROUPING(nc.SpecificOwnerType) = 0 THEN nc.SpecificOwnerType ELSE 'Total for ' + nc.BinnedOwnerType END AS SpecificOwnerType,
    COUNT(nc.CertLevel) AS TotalCertifications,
    ROUND(AVG(nc.NormalizedPointsAchieved), 2) AS AvgPointsAchieved,
    COUNT(CASE WHEN nc.CertLevel = '1' THEN 1 END) AS CertifiedCount,
    COUNT(CASE WHEN nc.CertLevel = '3' THEN 1 END) AS SilverCount,
    COUNT(CASE WHEN nc.CertLevel = '4' THEN 1 END) AS GoldCount,
    COUNT(CASE WHEN nc.CertLevel = '5' THEN 1 END) AS PlatinumCount
FROM 
    NormalizedCertifications nc
GROUP BY 
    ROLLUP(nc.BinnedOwnerType, nc.SpecificOwnerType)
HAVING 
    AVG(nc.NormalizedPointsAchieved) IS NOT NULL;


select * from DetailedOwnerTypeImpactWithTotals order by BinnedOwnerType, SpecificOwnerType;


CREATE VIEW OwnerOrganizationAnalysis AS
WITH NormalizedCertifications AS (
    SELECT 
        o.OwnerOrganization,
        cl.CertLevel,
        COUNT(p.ProjectID) AS ProjectCount,
        CASE 
            -- Normalize LEED v2 points
            WHEN cl.LeedVersion IN ('v2', 'v1') THEN 
                CASE 
                    WHEN cl.PointsAchieved BETWEEN 26 AND 32 THEN (cl.PointsAchieved - 26) / (32 - 26) * (49 - 40) + 40
                    WHEN cl.PointsAchieved BETWEEN 33 AND 38 THEN (cl.PointsAchieved - 33) / (38 - 33) * (59 - 50) + 50
                    WHEN cl.PointsAchieved BETWEEN 39 AND 51 THEN (cl.PointsAchieved - 39) / (51 - 39) * (79 - 60) + 60
                    WHEN cl.PointsAchieved BETWEEN 52 AND 69 THEN (cl.PointsAchieved - 52) / (69 - 52) * (110 - 80) + 80
                    ELSE cl.PointsAchieved  -- If points don't fall into expected ranges, leave them as-is
                END
            ELSE 
                cl.PointsAchieved  -- For LEED v3 and v4, leave points as they are
        END AS NormalizedPointsAchieved
    FROM 
        Projects p
    JOIN 
        Certifications cl ON p.ProjectID = cl.ProjectID
    JOIN 
        Owners o ON p.OwnerID = o.OwnerID
    WHERE 
        p.Country = 'US'
    GROUP BY 
        o.OwnerOrganization, cl.CertLevel, cl.LeedVersion, cl.PointsAchieved
)
SELECT 
    nc.OwnerOrganization,
    COUNT(nc.CertLevel) AS TotalCertifications,
    COUNT(DISTINCT nc.ProjectCount) AS DistinctProjects,
    ROUND(AVG(nc.NormalizedPointsAchieved), 2) AS AvgPointsAchieved,
    COUNT(CASE WHEN nc.CertLevel = '1' THEN 1 END) AS CertifiedCount,
    COUNT(CASE WHEN nc.CertLevel = '3' THEN 1 END) AS SilverCount,
    COUNT(CASE WHEN nc.CertLevel = '4' THEN 1 END) AS GoldCount,
    COUNT(CASE WHEN nc.CertLevel = '5' THEN 1 END) AS PlatinumCount
FROM 
    NormalizedCertifications nc
GROUP BY 
    nc.OwnerOrganization
HAVING 
    COUNT(nc.CertLevel) > 1;  -- Only include organizations with more than one certification

select * from OwnerOrganizationAnalysis order by TotalCertifications DESC, AvgPointsAchieved DESC;