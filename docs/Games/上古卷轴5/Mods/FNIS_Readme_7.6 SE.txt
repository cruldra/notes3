  Nexus  Skyrim File of the month May 2015]

No custom animation possible for Skyrim? Wrong. FNIS Behaviors allows other mods to add different types of animations to the game: idles/poses, sequenced, arm offset, furniture, and paired animations, killmoves, and creature animations.
And, with the demonstration mod FNIS Spells the user has a means to display almost all animation files. Dance animations included.

Note: You will have to run a generator tool GenerateFNISforUsers.exe (part of FNIS Behavior) every time you have installed or uninstalled FNIS, or an FNIS based mod. In addition, when you uninstall a mod that uses FNIS Creatures, you first have press the "De-Install Creatures" button, before you run the "Update FNIS Behaviors"

Use animations exclusive to the player in an EASY way? Install [url=http://www.nexusmods.com/skyrimspecialedition/mods/13281]FNIS PCEA2[/url] and see the potential
Bored of all females walking the same? [url=http://www.nexusmods.com/skyrimspecialedition/mods/13303]FNIS Sexy Move[/url] will change that
Ready to explore Skyrim from above? Install [url=http://www.nexusmods.com/skyrimspecialedition/mods/13322]FNIS Flyer[/url] and off you go
Want to see creatures animated by custom FNIS behaviors? Go and get [url=http://www.nexusmods.com/skyrimspecialedition/mods/?????]FNIS Zoo[/url] (NOT AVAILABLE YET for SSE)
You are a modder, and want to understand the contents of behavior files? [url=http://www.nexusmods.com/skyrim/mods/32113]CondenseBehavior[/url] represents Behaviors in a condensed form
You are a modder, and interested in Behaviors? Download FNIS for Modders Documentation V6.2 (or later) from the (FNIS for Skyrim) Files section 


[youtube]sYEAnTdZb-s[/youtube]
NEW: How to Use FNIS SE & LE | w/ MO2 & Vortex. The most complete FNIS tutorial. Thanks GamerPoets

More videos see at the end of this description


___________________________________________________________________________________
Recent Change(s)

2020/02/18
FNIS Behavior 7.6 SE, SE XXL, and VR XXL
-Recognize newest XPMSE skeleton bones (126/156) for [url=https://www.nexusmods.com/skyrimspecialedition/mods/1988]XP32 Maximum Skeleton Special Extended - XPMSSE[/url] Version 4.60 and above
-Allow future SSE skeleton versions to be recognized without the need of FNIS changes (based on skeleton.xml provided by mod)
-Increase animation number for FNIS XXL to 32.000
-Fixed a bug that prevented the PSCD patch
-Reduced output about Load CTD Calculation (see Note)
-Creature Pack: fix exit animation event names (capitalization, FrostBiteSpider + Mudcrab)

Note: Since version 7.5 FNIS calculates how many percent of Skyrim's custom animation resources you have used up, and will warn you when you have reached at least 99% of the load which causes CTD. In addition, FNIS XXL will show how much each of your animation mods contributes to Load CTD. However, since this FNIS functionality was implemented, there are several SKSE plugins available which circumvent this Skyrim bug. Which means that the problem can be easily avoided by users of many acustom animations. But since new users will not know about the fix, FNIS will still warn when the original limit is passed. The SKSE plugins that increase the Skyrim animation limit are:
-[url=https://www.nexusmods.com/skyrimspecialedition/mods/31146]Animation Limit Crash Fix SSE[/url] (Skyrim SE)
-[url=https://www.nexusmods.com/skyrimspecialedition/mods/17230]SSE Engine Fixes[/url] (Skyrim SE)
-[url=https://www.nexusmods.com/skyrim/mods/100672]Animation Limit Crash Fix LE[/url] (Skyrim LE)


___________________________________________________________________________________
Installation and Execution

NOTE: It is GENERALLY and STRONGLY recommended, that you DO NOT INSTALL any of Steam, Skyrim, FNIS in folders that are protected by Windows UAC (User Account Control). These are "C:/Program Files (x86)" or "C:/Program Files" (or their localized correspondents, e.g. "C:/Programme (x86)"). Use root folders like "D:\Games". This way you can avoid serious protection issues, especially when using Mod Managers.

FNIS is a tool, and not a mod, and therefore different from most everything that you find on Nexus. If you are new to modding and using FNIS, it is strongly recommended that you watch one of many videos that explain the use of FNIS. Like the videos at the beginning and the end of this description.


-Install FNIS Behavior SE V7_6
-Install Creature Pack V7_0 (optional, necessary for creature animation mods) 
-Install FNIS Idle Spells SE V5_0_1 (optional, necessary for the spells) 
-Activate the FNIS plugin (FNIS.esp)
-Install other FNIS dependant mods (see mod list below)
-
-If you install manually: Go to to your Skyrim Installation directory(for example D:/Games/Steam/SteamApps/common/Skyrim Special Edition), and from there to Data/tools/GenerateFNIS_for_Users
-If you install manually: Start the FNIS generator GenerateFNISforUsers.exe AS ADMINISTRATOR (part of FNIS Behavior SE, and ABSOLUTELY NECESSARY, or NOTHING works) 
-
-If you use NMM: Install FNIS like any other mod
-If you use NMM: If not done yet, "Configure FNIS" in NMM's list of supported tools (right most drop-down button in the menu bar). In the folder selection box go to your Skyrim Installation directory(for example D:/Games/Steam/SteamApps/common/Skyrim Special Edition), and from there to Data/tools/GenerateFNIS_for_Users. Click OK.
-If you use NMM: "Launch FNIS" from NMM's list of supported tools (right most drop-down button in the menu bar)
-
-If you use MO: Configure FNIS as executable (IMPORTANT - See one of the videos shown here, or read [url=http://wiki.step-project.com/Fores_New_Idles_in_Skyrim]S.T.E.P. Project: Fores New Idles in Skyrim[/url])
-
-Select necessary "Available Patches" (ONLY those you need! See section "Patches") from the bottom part of the generator window 
-Click "Update FNIS Behavior"
-
-To check if new animation mods can cause problems: press the "Consistence Check" button
-If you have uninstalled any mod that uses the FNIS Creature Pack: press the "De-Install Creatures"  button (and install the Creature Pack again, if necessary)


When you exit  GenerateFNISforUsers.exe, you  will be asked if a shortcut to the tool shall be created on your desktop.
Say "YES" when installing MANUALLY (STRONGLY recommended - you will need the generator quite frequently)
Say "Abort" when installing through NMM or MO


-For  more detailed manual installation directions see "Installation (extended)"
-For Installation under [url=http://www.nexusmods.com/skyrim/mods/1334]Mod Organizer[/url] see the guide at [url=http://wiki.step-project.com/Fores_New_Idles_in_Skyrim]S.T.E.P. Project: Fores New Idles in Skyrim[/url]. Or watch one of the videos shown here.
-If you are new with installing mods, check the Wiki [url=http://wiki.tesnexus.com/index.php/Installing_Mods_Using_NMM]Installing Mods Using NMM[/url]

_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
FNIS Vortex/MO support (profiles), FNIS.ini

FNIS supports Vortex and MO with several features. All features use parameters either from the FNIS.ini file, or as command line parameters when used from the command line. See also FNIS.ini0 in your FNIS Behaviors download.

File Redirection. This allows you to redirect all relevant FNIS generated files into a folder outside Skyrim\Data (to support Vortex/MO profiles). It includes all files that change the Skyrim animation functionality. FNIS temporary files and logs will still be written to the regular places (under data\tools\GenerateFNIS_for_Users). To use this functionality, the following parameter needs to be defined (FNIS.ini, or command line)
-RedirectFiles=<folder outside of Skyrim\Data>  e.g. RedirectFiles=D:\FNIS_Redirect To activate File Redirection, you need to set the patch for "File Redirection (Vortex/MO profiles support)". If you start FNIS from command line, this patch is automatically implied. To see how this can be used together with your actual mod manager, refer to your mod manager's documentation. 

Start FNIS from command line (done by Vortex, or from a user defined .bat file (not possible whith "virtual install"!). The commandline has the following format
-<FNIS_path>\GenerateFNISforUsers <FNIS_ini_parameters> e.g C:\Steam\Steamapps\common\Skyrim Special Edition\tools\GenerateFNIS_for_Users\GenerateFNISforUsers InstantExecute=1 PSCD=0 The <FNIS_ini_parameters> are the same that can be specified in the FNIS.ini file. (see FNIS.ini0)

FNIS execution without GUI (FNIS window). FNIS can be started without opening the FNIS window. The parameters (patches) are the same as in the last FNIS run.The FNIS window will only pop up if FNIS has to report error or warnings. This execution mode is activated with the following parameter (FNIS.ini, or command line)
-InstantExecute=1 When you use this feature manually (not via Vortex), it is recommended that you use it only from a .bat file (and not for example by double-clicking from the Explorer). Because otherwise you will not see when FNIS is done.

IMPORTANT Notes:
-All possible parameters are listed and explained in Skyrim\Data\FNIS.ini0. Empty lines, and lines starting with ' (apostrophe) are ignored
-To take effect, this file (FNIS.ini0) has to be copied MANUALLY to Skyrim\Data\FNIS.ini.
-Vortex, MO and NMM Virtual Install users: FNIS.ini has to go to "real" SKYRIM\Data (NOT inside the virtual mod folders!) 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
Problem shooting "Installation Problems"

When all NPCs move with the dreaded STIFF "half-T" POSE, then this is NO FNIS bug: YOU HAVE DONE SOMETHING WRONG DURING INSTALLATION!
-Remove the desktop GenerateFNISforUsers.exe desktop short-cut
-Re-install FNIS Behavior EXACTLY as described in the "Installation"
-Fix all errors in the generator's text field, and give note to all warnings - Errors definitely WILL, Warnings CAN indicate causes of the "half-T" problem
-DON'T try to be smart moving things around
-DON'T call the generator from within 7Zip, WinRar, ..., and NOT from a folder that you arbitrarily find through the Windows Explorer Search function


If the problem persists AND you are using a LEGAL, STEAM based version of Skyrim then report your problem WITH SUFFICIENT and PRECISE information in the FNIS discussion thread. Always include the contents of the generator's text field (copy/paste) from your last run. And the way you installed FNIS (NMM,MO, manual). And avoid meaningless information like "glitches", "didn't work", "tried all different ways 3 times".

_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
Piracy Precautions

FNIS takes preemptive steps against piracy:
-FNIS checks whether you use a legal Skyrim Steam installation. For this purpose reads the information Steam is setting every time you launch Skyrim with the Steam launcher. If this check fails, you will get a warning in the FNIS output window.
-FNIS checks whether you  are user of the mod pirate site "moddrop". In this case FNIS will abort with an appropriate error message. If you want to use FNIS again, you need to (1) uninstall the moddrop client, (2) uninstall FNIS, including all leftovers in tools\GenerateFNIS_for_Users or MO overwrites, respecively (3) re-install FNIS again.

NOTE:
-I will NOT HELP any PIRATES! First, I don't support people that don't support Skyrim and its mod authors. And second, I don't waste my time chasing down problems which only occur with illegal installations.
-In order to execute its preventive piracy actions, FNIS reads data from your PC. But only as much as is needed to find out the above described piracy information. If you don't fail the check, there is nothing that is kept in any way. And if you fail, the result is only distributed if you choose to share it. In particular there is no background scanning in any way.

You can use FNIS only if you agree with its piracy checks described above!!!

___________________________________________________________________________________
Skyrim SE Mods using FNIS Behavior SE

Non-adult:
-FNIS Spells SE (optional file for this mod)
-[url=http://www.nexusmods.com/skyrimspecialedition/mods/4668]Throwing Weapons Lite SE[/url] by JZBai
-[url=http://www.nexusmods.com/skyrimspecialedition/mods/4652]Animated Eating Redux SE editon[/url] by FlipDeezy
-[url=http://www.nexusmods.com/skyrimspecialedition/mods/4953]Cannabis Skyrim SE[/url] by MadNuttah and Virakotxa 
-[url=http://www.nexusmods.com/skyrimspecialedition/mods/10483]True Spear Combat - Weapons and Animations[/url] by Bouboule _ Marlborojack _ DemongelRex
-[url=http://www.nexusmods.com/skyrimspecialedition/mods/11734]Ride Sharing[/url] by Musje
-[url=http://www.nexusmods.com/skyrimspecialedition/mods/12848]Companion Selene Kate - Kiss Me[/url] by Kasprutz and Hello Santa
-[url=http://www.nexusmods.com/skyrimspecialedition/mods/13061]Pipe Smoking SE[/url] by MadNuttah and Virakotxa 
-[url=http://www.nexusmods.com/skyrimspecialedition/mods/13281]FNIS PCEA2 SE[/url] by fore
-[url=http://www.nexusmods.com/skyrimspecialedition/mods/13303]FNIS Sexy Move SE[/url] by fore
-[url=http://www.nexusmods.com/skyrimspecialedition/mods/13322]FNIS Flyer SE[/url] by fore
-[url=http://www.nexusmods.com/skyrimspecialedition/mods/?????]FNIS Zoo SE[/url] by fore coming soon
-[url=http://www.nexusmods.com/skyrimspecialedition/mods/13905]Flying Mod Beta and Flying Mod Overhaul (Converted for SSE)[/url] by porroone and Indefiance
-[url=http://www.nexusmods.com/skyrimspecialedition/mods/14083]Shake It SSE - Dance Animations Full Version 4.5[/url] by indexMemories
-[url=http://www.nexusmods.com/skyrimspecialedition/mods/14775]Play Random Idle SE[/url] by Phnx
-[url=http://www.nexusmods.com/skyrimspecialedition/mods/14973]Potion Animated fix (SE)[/url] by migeandend
-[url=http://www.nexusmods.com/skyrimspecialedition/mods/15309]TK Dodge SE[/url] by tktk1
-[url=http://www.nexusmods.com/skyrimspecialedition/mods/15326]a little sexy apparel replacer with LSAR Addon[/url] by Cotyounoyume
-[url=http://www.nexusmods.com/skyrimspecialedition/mods/15394]New Armoury - Rapiers with third person animations SSE Version[/url] by NickNak
-[url=http://www.nexusmods.com/skyrimspecialedition/mods/15452]Crawl on all Fours animation for SSE[/url] by DarkAngel1265

Adult:
-[url=http://www.nexusmods.com/skyrimspecialedition/mods/189]Maids II - Deception[/url] by Enter_77 
-[url=http://www.nexusmods.com/skyrimspecialedition/mods/5618]Animated Prostitution SE[/url] by Xider 
-[url=http://www.nexusmods.com/skyrimspecialedition/mods/5668]Music and Dancing in the Luxury Suite SE[/url] by Migal130 
-[url=http://www.nexusmods.com/skyrimspecialedition/mods/5941]Flower Girls SE x[/url] by Xider 
-[url=http://www.nexusmods.com/skyrimspecialedition/mods/12832]EstrusForSkyrimSE[/url] by Cotyounoyume
-[url=http://www.nexusmods.com/skyrimspecialedition/mods/13207]CBP Physics[/url] by Polygonhell

Mods supported by FNIS with patches:
-[url=http://www.nexusmods.com/skyrimspecialedition/mods/15309]TK Dodge SE[/url] by tktk1

___________________________________________________________________________________
Usage

The main file of FNIS, FNIS Behaviors, is a tool, NOT a mod. Actually 2 tools. Mod authors can use the modders tool to generate behavior files to add new custom animations to their own mods. The other one, the users tool, then will collect the data for all FNIS dependent tools on the user PC and then integrate that into the user's Skyrim installation. 

To hook new animations into the sytem, it is necessary to modify original ("vanilla") behavior files for characters and creatures. Because of the anti-modding structure of these behaviors there cannot be 2 mods that modify the same behavior files.

Mods that use FNIS are providing animation lists, simple text files that tell FNIS with a variety of parametwers, how to introduce the animations into Skyrim. The FNIS user tool then collects all this data and creates ONE set of modified behavior files which have all the information needed. No mod author has to touch behavior files, and no animation mod is incompatible with the others. Provided that they all use FNIS as their framework. 

FNIS animations can be used for all kind of animation tasks, ranging from simple idles to idles sequences, furniture animations, paired animations, killmoves, and more. FNIS also allow to dynamically replace all the most important standard animations, like movement, combat, 

(For more modding related information see FNIS for Modders Documentation in the files section)


_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
Patches

To make 2 behavior modifying mods compatible, they have to be integrated into one set of behaviors.  That's why [color=CHOCOLATE]GenerateFNISforUsers.exe has an integrated "Patch Management", which takes care of necessary Behavior changes for other behavior changing mods, like PC Exclusive Animation Path. However, FNIS will not include other (animation) files from these mods. So if you want to use the patches, you will still have to install these mods as well, before running the generator.

DONT' TICK PATCHES for mods, that you DON'T HAVE INSTALLED!!

Note that FNIS doesn't need to add patches for modes that "only want to add custom animations via FNIS. FNIS will automatically find those mods (if properly installed), and list them in the generator output.

There are 3 more patches without required base mods:
-SKELETON arm fix: this change (originally introduced by mirap for CHSBHC) fixes arm animation glitches in many animations, caused by the additional bones in custom skeletons. This patch checks for the actually installed skeletons (female and male), and applies skeleton specific modifications to the behavior files.
-GENDER specific animations: in Skyrim only a few animations are defined gender specific (different animations in the "Animations/female" and "Animations/male" folder. This patch allows you to make other animations gender specific provided they are "properly" created by the animators, see the Note). If you want to use such animations (e.g  dualsun's [url=http://www.nexusmods.com/skyrimspecialedition/mods/3761]Pretty Combat Idles SSE[/url]), simply copy them into your "Animations/female" (or "male"), and use this patch.
-HKX File Compatibility Check Skyrim/SSE: Cross-using behavior, skeleton, and animation files between Skyrim and Skyrim SE will result in the well-known t-pose. General t-pose when behavior or skeleton files from the other Skyrim are used, animation-long t-pose in case of cross-used animations. This patch will check for such bad files. Obviously, it only makes sense to use this patch if you experience t-pose situations. And FNIS will not keep this patch ticked for more than one FNIS generation.


Note (for Skyrim SE only!): Most mods that have been patched by FNIS in Skyrim 32 bit have not been ported to SSE yet. The FNIS generator for SE is listing these mods with "NOT AVAILABLE for SSE". This way you can immediately use them without a new FNIS update when they come out. Or in case you decide to port them yourself. Usually this is not very hard. 


_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
Bug Fixes

FNIS Behaviors incorporate the following bug fixes for vanilla Skyrim (without the need for the user to actively do anything):
-Death after hug fix: a character (player or NPC) drops dead after executing the paired hug animation (paired_huga.hkx). This death happens when the character "should" have died before and has gone into the bleedout state after a brawl, or because s/he is essential.
-[url=http://www.nexusmods.com/skyrim/mods/68794]Proper Spell Cast Direction[/url] fix by Staff of Flames: a person bends his spine backwards when he is aiming towards something flying in the sky(a dragon, for example). This natural posture happens to all NPCs in Skyrim and to the player when he is using a bow, but not when the player is using magic. NOTE 1: FNIS implements PSCD WITHOUT arm, hand and neck twist as it is included in the PSCD mod. Certain types of animations, including bow and animobject animations don't work well with this type of twist. NOTE 2: Some users do not appreciate the amount of bending the player with this patch. By setting "PSCD=1" in the FNIS.ini you can activate this bug fix. Read the file FNIS.ini0 and follow the instructions there.
-[url=http://www.nexusmods.com/skyrim/mods/36453]Dual Dagger Power Attack Speed Fix[/url] by Alek: fixes dual power attack speed scaling when using daggers, which will now be affected by Dual Flurry perks, Elemental Fury shout or any other kind of attack speed buffs.



_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
FNIS and FNIS XXL, Animation Load CTD Limit

FNIS comes in 2 versions:
-FNIS, which allows to add 10,000 animations for character 3rd person, AND for each of Skyrims creatures. That is plenty for most of the users of animation mods.
-FNIS XXL, which allows to add 32,000 animations for race. For the animation collectors amongst us.
However there is a bug in Skyrim that will cause CTD (Crash to Desktop) when starting the game with too many animations added. This happens usually somewhere in the range between 11k and 14k additional 3rd person character animation, depending on the type of animations. (1st person, animation replacers, and creature animations are not affected).

FNIS (both versions) calculates how many percent of Skyrim's custom animation resources you have used up, and will warn you when you have reached at least 99% of the load which causes CTD. In addition, FNIS XXL will show how much each of your animation mods contributes to Load CTD. For more information: see the FNIS SE Article [url=https://www.nexusmods.com/skyrimspecialedition/articles/1168]Skyrim Animation Load CTD: Understand the background of this unpleasant restriction. And how FNIS can help you.[/url]

However, since this FNIS functionality was implemented, there are several SKSE plugins available which circumvent this Skyrim bug. Which means that the problem can be easily circumvented by users of many acustom animations. But since new users will not know about the fix, FNIS will still warn when the original limit is passed. The SKSE plugins that increase the Skyrim animation limit are:
-[url=https://www.nexusmods.com/skyrimspecialedition/mods/31146]Animation Limit Crash Fix SSE[/url] (Skyrim SE)
-[url=https://www.nexusmods.com/skyrimspecialedition/mods/17230]SSE Engine Fixes[/url] (Skyrim SE)
-[url=https://www.nexusmods.com/skyrim/mods/100672]Animation Limit Crash Fix LE[/url] (Skyrim LE)


___________________________________________________________________________________
FNIS Spells

Gamers can use it to make characters perform any standard or custom idle by simply copying the (.hkx) animation file under a pre-defined name into a special directory, and activate them by casting the FNIS spells. Note: although there can be different NPCs performing different animations at one time, this mod does not provide any synchronization of these animations, as needed for example in some adult type animation mods.

On start-up the player automatically receives 3 spells:

-FNIS Spell Selector: This (self) spell allows you to select 1 of 72 Idle Animation which will be used for the next Idle Spells (NPC or player). By installation, most of these animations are pre-set with standard animations, but it also includes all 17 of Umpa's [url=http://www.nexusmods.com/skyrim/mods/2658]Dance Animation for Modder[/url] (animations C10 to C26), and dualsun's [url=http://www.nexusmods.com/skyrim/mods/17880]Funny Animations And Idles[/url] (animations C28 to C34, A1 to A4). For a complete list of the used spells, see [color=salmon]Data/meshes/actors/character/animations/FNISSpells/FNIS_Animations.txt.
-FNIS NPC Idles: This (target) spell makes the casted NPC perform the selected idle. On standing NPCs the idle will run until you interrupt by casting again. But unfortunately, in some cases the animation can be also interrupted by or blended with the movement of other standard animation of the NPC's regular animation schedule. 
-FNIS Player Idles: This (self) spell forces the player into 3rd person and makes her perform the selected idle. The will run until you interrupt, e.g. by jumping up (space bar).


If you want to cast animations of your choice, you can simply replace the files in meshes/actors/character/animations/FNISSpells by your own files:

-FNISSPc001.hkx to FNISSPc045.hkx (for cyclic animations)
-FNISSPa001.hkx to FNISSPa009.hkx (for acyclic animations)
-FNISSPo001.hkx to FNISSPo018.hkx (for cyclic animations using AnimObjects)
-FNISSPb001.hkx to FNISSPb009.hkx (for acyclic animations using AnimObjects)

If you change AnimObject ('o' or 'b') animations, you should also change the corresponding AnimObjects in meshes/AnimObjects/FNIS


___________________________________________________________________________________
FAQ

I'm getting Error messages when I call GenerateFNISforUSers.exe. What can I do?
All error codes greater than 2000 are logical errors found by the Generator. Errors less than 2000 are reported by Windows. Most of the reports should be self-explanatory. Before you can work with FNIS, you need to solve all errors shown in the textfield. Also look at the warnings. They can also indicate problems that prevent FNIS from working properly (like incompatibilities). Some specific error messages:
-Error 5 (Access denied): See the following FAQ entry
-Error 53 (File not found): You probably have an incomplete installation -> Re-Install FNIS Behavior. If the problem persist then then this can be caused by an AntiVirus program giving a fals positive (WebRoot and SuperAntiSpyware have caused a lot of trouble in the past). Deactivate your AV during FNIS installation.
-Error 2019 (Temporary directory does not exist. Bad Installation?): (1) You are running GenerateFNISforUSers.exe from a wrong place. Most likely from out of the archive file. DON'T do that. Or (2) you have an incomplete installation -> Download and Re-Install FNIS Behavior 


I'm getting "ERROR(5): Access to the path .... is denied" 
You have to run GenerateFNISforUSers.exe as Administrator. Right-click on the Generator (in in Explorer, or on the desktop link) and click "Run as Administrator". Or you can permanently set this option in the GenerateFNISforUsers.exe properties. GenerateFNISforUsers.exe -> Right-Mouse-Button -> Properties -> Compatibility -> Tick "Run as Administrator" (Windows 10).

I'm getting "ERROR(5): Access to the path .... is denied" DESPITE running as administrator
Recently there is a rising number of users who get this error trying to run FNIS. Usually (not always) these users use NMM, and have Steam, NMM, Skyrim all installed under "C:\Program Files(x86)".

This is NOT an FNIS issue. It has to do with authorization issues on your PC, and even the usual running as admin does not help. Usually it is NMM which creates the issue as you can see from DuskDweller's response posted as part of the known issue "ERROR(2001)" below. It seems to be caused by the Windows UAC (User Account Control), by which the Microsoft tries to protect "C:\Program Files(x86)" from the users wrong doing.
Since we have no idea what is causing this alarmingly increasing rate of reports, I ask you to give some more information about your environment: NMM/MO/manual install, Windows version, used AntiVirus, full generator output, more information from the Windows Event Viewer Eventvwr.msc, ...

To make FNIS work you can try to install FNIS manually. Remove "Data/tools/GenerateFNIS_for_Users" first. After installation go to GenerateFNISforUsers.exe -> Click Right-Mouse-Button -> Properties -> Compatibility -> Tick "Run as Administrator" -> OK.
The only know way to RELIABLY solve this issue is to move Steam and Skyrim away from "C:\Program Files(x86)". WHAT YOU GENERALLY SHOULD DO. For example to "C:\Games". Don't be afraid. This is not hard. See [url=http://support.steampowered.com/kb_article.php?p_faqid=231]Moving a Steam Installation and Games[/url]


("Animation Problems") After installing FNIS I suddenly have animations problems (stiff half-T pose, or other glitches). Why?
Principally, FNIS Behaviors correctly installed and generated (without patches) adds additional animations, and DOES NOT INFLUENCE OTHER ANIMATIONS IN ANY FORM! There are a numer of possible causes, which can make FNIS look like it is breaking animations:
-You have ignored ERRORS or WARNINGS in your last generator run. Don't do this. They are there for some reason.
-You are running a cracked Skyrim. In the past apparant cracks have shown symptoms I have not seen in Steam versions.
-Your game cache is corrupted. Run Steam Verify Cache (The Elder Scrolls V: Skyrim -> RMB -> Properties -> Local Files -> Verify Integrity Game Cache ...). Another reason not to use cracks.
-You are running a custom skeleton which comes with a skeleton.nif only, and not a corresponding skeleton.hkx. Installing [url=http://www.nexusmods.com/skyrimspecialedition/mods/1988]XP32 Maximum Skeleton Special Extended - XPMSE[/url] first usually should fix the problem.

I got the warning "Generator not run from a legal (Steam) Skyrim installation directory. If you use SKSE, this can be fixed by starting Skyrim once through Steam."
This is a warning (only) that indicates, that your installation is not recognized as a legal steam installation.  The generator is not effected by this. However, I will NOT support you with any installation problem you might have. Most users seeing this warning are using an illegal or pirated Skyrim copy (and don't have the proper registry information). And I don't support Skyrim pirates, and I don't want to track down possible problems caused by pirate installations.
If you think you got a legal installation
-check Bethesda's [url=http://forums.bethsoft.com/topic/1258133-installing-from-disc-instead-of-steam]installing from disc instead of steam[/url] thread.
-you might not have executed the proper steps to relocate your steam folder, as described in [url=http://support.steampowered.com/kb_article.php?p_faqid=231]Moving a Steam Installation and Games[/url].
-if you have used "Steam Verify Cache", it might have removed one of the required registry entries. Start Skyrim once through the "Skyrim Launch" (and not through SKSE)
-or, most likely, a hyperactive registry cleaner (CCLeaner?) has removed the registry entries used for this check

("Load CTD") After I ran the generator, Skyrim aborts (CTD) when loading a save file. An FNIS bug?
No. Skyrim's memory Management is buggy, and the more you add to Skyrim, the greater the likelyhood that you see CTDs, especially on save load. Here my advice when you have an unacceptable CTD rate:
-Install [url=http://skse.silverlock.org]SKSE 1.7.0[/url] or later. This will install "sheson's memory patch", which will not totally cure Skyrim, but geatly reduce CTD likelyhood. Make sure you have a Data/SKSE/skse.ini file with the following lines: 
[Memory]
DefaultHeapInitialAllocMB=768
ScrapHeapSizeMB=256
-Use one of those SKSE plugins: [url=https://www.nexusmods.com/skyrimspecialedition/mods/31146]Animation Limit Crash Fix SSE[/url] (Skyrim SE), [url=https://www.nexusmods.com/skyrimspecialedition/mods/17230]SSE Engine Fixes[/url] (Skyrim SE), [url=https://www.nexusmods.com/skyrim/mods/100672]Animation Limit Crash Fix LE[/url] (Skyrim LE)

(".NET Problems") The generator gives some other strange error messages like "0x800A0005 (CTL_E_ILLEGALFUNCTIONCALL)", ".NET Framework Initialization Error", or aborts immediately after double-clicking.
Most likely you have a corrupted Windows or .NET installation, which can even be caused by undetected viruses. Use the Windows Event Viewer [url=http://windows.microsoft.com/en-us/windows/what-information-event-logs-event-viewer#1TC=windows-7]eventvwr.msc[/url] to find clues about the cause of the failure. If it indicates .NET problems, then re-install .NET from [url=http://www.microsoft.com/en-us/download/details.aspx?id=17718]Microsoft .NET Framework 4 (Standalone Installer)[/url]. If this does not help use [url=http://blogs.msdn.com/b/astebner/archive/2008/10/13/8999004.aspx].NET Framework Setup Verification Tool[/url] and [url=http://www.microsoft.com/en-us/download/details.aspx?id=30135]Microsoft .NET Framework Repair Tool[/url]. The following links might give additional information: [url=http://social.msdn.microsoft.com/Forums/en/netfxsetup/thread/3d03a6bd-d7ac-4972-bc22-88fc016d7e63].NET 40 Redistib install error "failed with 0x240006"[/url] and [url=http://blogs.msdn.com/b/astebner/archive/2010/12/29/10110053.aspx]Possible issue where .NET Framework 4 setup reports success but fails ...[/url]. This [url=http://blogs.msdn.com/b/astebner/archive/2006/09/04/739820.aspx].NET Framework Setup Verification Tool User's Guide[/url] seemed to solve many issues. If nothing helps, and before you have to completely re-install windows, you might consider [url=http://support.microsoft.com/kb/2255099]How to Perform an In-Place Upgrade on Windows Vista, Windows 7, Windows Server 2008 & Windows Server 2008 R2[/url].

What is a "clean save" (or "cleaned save"), and how can I make one? 
There are (hardly understood) mechanisms in the engine which cause problems with updates of scripted mods. The mixture of old save data, new scripts, quests, and objects can cause all kinds of problems. When it does not help to use another save, where FNIS Spells have not been used during the last 3 minutes, you should try with a "clean save"

-Dismiss any follower
-Go to a vanilla interior cell
-Equip only vanilla equipment
-Save and Quit
-Deactivate all pugins that you want to update (or, all except Skyrim.esm and Update.esm)
-Run, load the previous save, save again -> THIS is your "clean save"
-Activate your plugins

There are more complicated "clean save" procedures reported, but they might only be necessary for more complicated mods

I have uninstalled one of the mods which use FNIS Behavior. Now I get CTD when loading a save file. 
That's a sytem reaction when a behavior file is missing. Run GenerateFNISforUsers.exe and you should be fine again.

Starting with FNIS 6.0 there is an FNIS.esp. Do I really need this? And if so, where do I put it in the load order.
FNIS.esp is needed when you have at least one mod using the FNIS Alternate Animation (AA) functionality, like in XPMSE, PCEA2, or Sexy Move. But even if you don't use one of these, I don't recommend deactivating FNIS.esp. Because if you do, and if you THEN add such mods, you will certainly not know why all of the sudden things don't work. And you can put FNIS.esp first in your list. Though it doesn't really matter. It doesn't conflict with anything, and is only needed once after game load.

I want to run the Generator on Linux, but it cannot be run there because of .NET 4.x. 
Here is what you can do (thanks to hellgeist):
1- On any version of Linux, install VirtualBox, put windows XP on it
2- Update windows and Install .Net 4
3- Install Skyrim and patch it via Steam. 
4- Use VMWare Tools to add a shared drive so you can port files between Linux and XP.
5- Copy the Data folder from your main install on linux and overwrite the windows one.
6- Install the ForesNewIdles in Skyim-FINIS mod, and run GenerateFNISforUsers.exe
7- Make sure your main Linux data folder is backed up.
8- Copy the updated Data folder from VirtualBox (Windows) Skyrim and overwrite your main one on Linux
9- Close VirtualBox and run Skyrim.

I'm a modder and have a mod which conflicts with the behavior files modified by FNIS. How can I be compatible with FNIS? 
Talk to me. I can add your behavior changes as patches to the FNIS generator. However, I have ONE big requirement: I'm not going to figure out what your changes are compared to the vanilla xml file. That can be a verytedious task. I need EXACTLY what you changed in the vanilla file (similar to a UNIX diff). 

___________________________________________________________________________________
Manual INSTALLATION (extended)

Requirements:
-Microsoft .NET 4.0 (follow the link when  GenerateFNISforUsers.exe issues a corresponding error message)
-Administrator rights to your Skyrim installation directories

Note: there are many ways to manually install. This is "my way":

-Create a temporary directory <temp_dir> (name of your choice)
-Download FNIS Behavior SE V7_6 into <temp_dir>
-If you want to use Creature Animation mods: Download Creature Pack V7_0 into <temp_dir>
-Download FNIS Spells V5_0_1 -- ADD-ON for the spells into <temp_dir>
-Open and extract FNIS Spells 5.0.1.7z with your favorite archive program (e.g. 7zip) into into <temp_dir>
-Open and extract Creature Pack V7.6.7z into into <temp_dir>
-Open and extract FNIS Behavior 7.6.7z into into <temp_dir> - Say "Yes" when asked to "Integrate...". Now you should have a "<temp_dir>/Data" folder with 3 subfolders: "Meshes", "Tools", And "Scripts". 
-Drag and drop the <temp_dir>/Data folder into your Skyrim installation directory (usually [color=salmon]C:/Program Files (x86)/Steam/SteamApps/common/skyrimspecialedition). Say "Yes" when asked "Overwrite?", or when asked to "Integrate..."
-Install other FNIS dependant mods (see mod list above)
-Goto to your Skyrim Installation directory, and from there to [color=salmon]Data/tools/GenerateFNIS_for_Users
-Start (double-click) [img]http://www.mediafire.com/conv/1d7d11201fe82ed68231a5859d843d5b1fc9b4ebf5f446b77f64a4d2140f4e1f6g.jpg[/img]  GenerateFNISforUsers.exe (part of FNIS Behavior) ABSOLUTELY NECESSARY, or NOTHING works
-Select necessary Behavior patches in the Possible Patches field (make sure you have the patched mod installed!)
-Click Update FNIS Behavior (this will create your installation-specific behavior files)
-Click Behavior Consistence Check (optional, checks if new mods have defined inconsitent stuff)
-Click Exit. Say "Yes" when asked to "Create a shortcut on your desktop?" (strongly recommended, you will need the generator quite frequently)
-Start Steam and click "Play"
-In "Data Files" tick FNIS.esp
-In "Data Files" tick FNISSpells.esp


For NMM (Nexus Mod Manager) or MO (Mod Organizer) installation see the appropriate videos

___________________________________________________________________________________
UNINSTALL

Manual Uninstall

First, if you have the FNIS Creature Pack installed: Open the Generator and press "De-Install Creatures".
After that, remove the following files or directories:

-Data/FNIS.esp
-Data/FNISspell.esp
-Data/Meshes/actors/character/animations/FNIS
-Data/Meshes/actors/character/animations/FNISBase
-Data/Meshes/actors/character/animations/FNISSpells
-Data/Meshes/actors/character/behaviors/*.hkx
-Data/Meshes/actors/character/characters/defaultmale.hkx
-Data/Meshes/actors/character/characters female/defaultfemale.hkx
-Data/Meshes/animationsetdatasinglefile.txt
-Data/Meshes/animationdatasinglefile.txt
-Data/Meshes/AnimObjects/FNIS
-Data/Tools/GenerateFNIS_for_Modders
-Data/Tools/GenerateFNIS_for_Users


NMM Uninstall

First, if you have the FNIS Creature Pack installed: Open the Generator and press "De-Install Creatures".
Then deactivate both FNIS Behavior, FNIS Creature Pack, and FNIS Spells in NMM's Mods tab. That should take care of everything. To be totally sure that there are no behavior remnants, delete the files from the previous Manual Uninstall section, if they should still exist:
 
___________________________________________________________________________________
Known Bugs, Restrictions and Incompatibilities

-This mod is incompatible with every other mod which modifies behavior files (character or creatures).  However, when you are a modder and willing to cooperate, then I'm happy to add your behavior changes as patches into the FNIS generated behaviors. But beware, I need precise information about your changes, as delta to the original files. It really pays to document your changes in the behavior files from the beginning of your development. Like you can see it in FNIS' templates for the generated files.

-(FNIS Spells) It's a little tricky to end the player's idles. Reason is that the CK doesn't provide an event for hitting a key (for interruption). The easiest way to end a player idle is to use space to jump up, then toggle to 1st to re-gain casting ability. Note: this spell DOES NOT change your standard idle(s). It only makes the the player character play the idle until you interrupt.

-Few custom animations can behave "glitchy" when [b]TK Dodge 3.0[/b] is active. TK Dodge uses an SKSE plugin which modifies a file that was generated by the FNIS generator (AnimationDataSingleFile.txt). Unfortunately this modification has a bug which changes the status of 4 custom animations NOT added by TK Dodge (at position 2000 to 2003). Such modifications can cause sporadic t-pose, double-hit, pre-mature abort to such animation.


___________________________________________________________________________________
Usage for MODDERS

See FNIS for Modders Documentation in the files section.

A short syntax for Anim definition can be found at the top of each FNIS (character) AnimList (e.g. FNIS_FNISBase_List.txt)

___________________________________________________________________________________
History (major steps)

2016/11/06 V7.0 Beta . Initial Release (based on FNIS Behavior (Skyrim 32bit) V6.3)
2017/10/30 V7.0 . . Generator exe files working for both Skyrim/SE, animation number 10k/20k XXL, HKX File Compatibility Check Skyrim/SSE 
2018/02/17 V7.1 . . Added patch for TK Dodge SE and other tktk1 mods, added FNIS.ini functionality (switch off PSCD bug fix), bug fixes and installation checks
2018/02/23 V7.1.1 . Fixed load CTD bug TK Dodge patch with furniture animation mods, wrong Consitence Check warnings, Russian languge file update
2018/03/04 V7.2 . . Fixed some minor bugs, combined version with FNIS for Skyrim
2018/04/02 V7.3 . . Vortex and MO support release: Added File Redirection, Start from command line, Execution without GUI
2018/05/20 V7.4.5 . Important bug fixes for Users (avoid removal of unrelated files) and Modders (wrong transition parameters)
2019/08/23 V7.5 . . Introduced Load CTD Calculation
2020/02/18 V7.6 . . Final FNIS SE version




___________________________________________________________________________________
Credits

TheFigment aka The Hologram for his invaluable [url=http://www.nexusmods.com/skyrim/mods/1797]hkxcmd[/url]
Umpa for [url=http://www.nexusmods.com/skyrim/mods/2658]Dance Animation for Modder[/url]
dualsun for [url=http://www.nexusmods.com/skyrim/mods/17880]Funny Animations And Idles[/url]
mirap for the CHSBHC arm fix
cougarbg(Bulgarian), Zimitry(Español], latranchedepain(Français), speleologo(Italiano), vicpl(Polski), kapasov(Russian), xyzeratul (Chinese), fofaun1417/NomexPT(Português), sssSami(Norsk), 1stchannel(Indonesian), Luke2135(Hungarian), yohru/kuroko(Japanese), Xarec (Swedish), nyaamolip (Korean), Kavij (Serbian) for Generator translations
somebody4 for input, feedback and test for pre-cache functionality. And for pushing me in moments when I wanted to give up on this matter.

___________________________________________________________________________________
Licensing/Legal

The FNIS Behavior SE can only be downloaded and used in the described way. Without my express permission you are NOT ALLOWED
-to upload FNIS Behavior TO ANY OTHER SITE
-to distribute FNIS Behavior SE as part of another mod
-to distribute modified versions of FNIS Behavior
-to make money with files which are part of FNIS Behavior, or which are created with the help of FNIS Behavior

You can use and modify FNIS Idle Spells SE in any way that does not prevent running (the original) FNIS Idles Spells in its described way. Simply give credit, and inform me when you include it into your mod.


___________________________________________________________________________________
More Videos

[youtube]iukqiPz7XuU[/youtube]
The most complete FNIS tutorial for NMM and MO. Thanks GamerPoets

[youtube]Q7ZgIHWV5Ec[/youtube]
A little bit what FNIS is all about. Thank you, shinji72

[youtube]XLZ2OLK4k68[/youtube]
FNIS intallation instructions at 1:27, 10:37, and 11:41

[youtube]VpOuGN-yr6E[/youtube]
Using Mod Organizer to install FNIS. Thank you gopher for both
