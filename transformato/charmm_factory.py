import datetime

def charmm_factory(configuration: dict, structure: str, env: str):
    """Function to build the string needed to create a CHARMM input and streaming file"""

    if env == "vacuum":
        env_dir = configuration["system"][structure]["vacuum"]["intermediate-filename"]
    elif env == "waterbox":
        env_dir = configuration["system"][structure]["waterbox"]["intermediate-filename"]

    #tlc = configuration["system"][structure]["tlc"]
    nstep = configuration["simulation"]["parameters"]["nstep"]
    nstout = configuration["simulation"]["parameters"]["nstout"]
    nstdcd = configuration["simulation"]["parameters"]["nstdcd"]
    steps_for_equilibration = configuration["solvation"]["steps_for_equilibration"]
    #switch = configuration["simulation"]["parameters"]["switch"]
    #GPU = configuration["simulation"]["GPU"]
    try:
        GPU = configuration["simulation"]["GPU"]
    except KeyError:
        GPU = False
        pass
    try:
        switch = configuration["simulation"]["parameters"]["switch"]
    except KeyError:
        switch = "vswitch"
        pass

    # building whole file
    if env == "vacuum":
        charmm_vacuum = charmm_string(
            env, env_dir, nstep, nstout, nstdcd, steps_for_equilibration, switch, GPU
        )
        return charmm_vacuum
    elif env == "waterbox":
        charmm_waterbox = charmm_string(
            env, env_dir, nstep, nstout, nstdcd, steps_for_equilibration, switch, GPU
        )
        return charmm_waterbox

# toppar file
def build_reduced_toppar(tlc):
    date = datetime.date.today()
    toppar = f"""* Simplified toppar script 
* Version from {date} 
*

! Read Protein Topology and Parameter 
open read card unit 10 name ./toppar/top_all36_prot.rtf 
read  rtf card unit 10 
    
open read card unit 20 name ./toppar/par_all36m_prot.prm 
read para card unit 20 flex 

! Read Nucleic Acids 
open read card unit 10 name ./toppar/top_all36_na.rtf 
read  rtf card unit 10 append 
    
open read card unit 20 name ./toppar/par_all36_na.prm 
read para card unit 20 append flex
    
! Read Carbohydrates 
open read card unit 10 name ./toppar/top_all36_carb.rtf 
read  rtf card unit 10 append 
    
open read card unit 20 name ./toppar/par_all36_carb.prm 
read para card unit 20 append flex 

! Read Lipids 
open read card unit 10 name ./toppar/top_all36_lipid.rtf 
read  rtf card unit 10 append 
    
open read card unit 20 name ./toppar/par_all36_lipid.prm 
read para card unit 20 append flex
    
!Read CGENFF 
open read card unit 10 name ./toppar/top_all36_cgenff.rtf 
read  rtf card unit 10 append 
    
open read card unit 20 name ./toppar/par_all36_cgenff.prm 
read para card unit 20 append flex
    
! Additional topologies and parameters for water and ions 
stream ./toppar/toppar_water_ions.str

! Read {tlc} RTF 
open read unit 10 card name {tlc}_g.rtf 
read rtf card unit 10 append

! Read {tlc} prm 
open read unit 10 card name {tlc}.prm 
read para card unit 10 append flex

! Read dummy_atom RTF 
open read unit 10 card name dummy_atom_definitions.rtf 
read rtf card unit 10 append

! Read dummy_atom prm 
open read unit 10 card name dummy_parameters.prm 
read para card unit 10 append flex

"""
    return toppar


def charmm_string(
    env: str,
    env_dir: str,
    nstep: int,
    nstout: int,
    nstdcd: int,
    steps_for_equilibration: int,
    switch: str,
    GPU: bool,
):
    """Body of the CHARMM file with option for gas pahse, waterbox with vswitch and vfswitch"""
    if GPU == True:
        GPU = f"""domdec gpu only"""
    else:
        GPU = ""

    header = f"""*Version September 2020 
*Run script for CHARMM jobs from transformato 
*

! Read topology and parameter files 
stream charmm_toppar.str 

! Read PSF 
open read unit 10 card name {env_dir}.psf 
read psf  unit 10 card

! Read Coordinate 
open read unit 10 card name {env_dir}.crd 
read coor unit 10 card
"""
    ##### gas phase ######
    gas_phase = f"""
coor orie sele all end ! put the molecule at the origin

MMFP
GEO rcm sphere -
    Xref 0.0 Yref 0.0 Zref 0.0 XDIR 1.0 YDIR 1.0 ZDIR 1.0 -
    harmonic FORCE 1.0 select .not. ( hydrogen .or. resname TIP3 ) end
END

set ctofnb 990.
set ctonnb 980.
set cutnb  1000.

nbonds ctonnb @ctonnb ctofnb @ctofnb cutnb @cutnb -
  atom swit vatom vswitch -
  inbfrq 1 

energy   inbfrq 1
{GPU}
energy   inbfrq 0

mini sd nstep 100

set nstep = {nstep} 
set temp = 300.0

scalar fbeta set 5. sele all end
open write unit 12 card name charmm_gasp.rst
open write unit 21 file name charmm_gasp.dcd
 
DYNA lang leap start time 0.001 nstep @nstep -
    nprint {nstout} iprfrq {round(nstep/20)} -
    iunread -1 iunwri 12 iuncrd 21 iunvel -1 kunit -1 -
    nsavc {nstdcd} nsavv 0 -
    rbuf 0. tbath @temp ilbfrq 0  firstt @temp -
    echeck 0
    
stop"""

    ##### waterbox ######
    liquid_phase = f"""
!
! Setup PBC (Periodic Boundary Condition)
!

stream charmm_step3_pbcsetup.str

!
! Image Setup
!

open read unit 10 card name charmm_crystal_image.str
CRYSTAL DEFINE @XTLtype @A @B @C @alpha @beta @gamma
CRYSTAL READ UNIT 10 CARD

!Image centering by residue
IMAGE BYRESID XCEN @xcen YCEN @ycen ZCEN @zcen sele resname TIP3 end

!
! Nonbonded Options
!

nbonds atom vatom {switch} bycb -
       ctonnb 10.0 ctofnb 12.0 cutnb 16.0 cutim 16.0 -
       inbfrq -1 imgfrq -1 wmin 1.0 cdie eps 1.0 -
       ewald pmew fftx @fftx ffty @ffty fftz @fftz  kappa .34 spline order 6

energy
{GPU}
energy

!
!use a restraint to place center of mass of the molecules near the origin
!

MMFP
GEO rcm sphere -
    Xref @xcen Yref @ycen Zref @zcen XDIR 1.0 YDIR 1.0 ZDIR 1.0 -
    harmonic FORCE 1.0 select .not. ( hydrogen .or. resname TIP3 ) end
END

!
! NPT dynamics:
! you can change
! nstep  : number of MD steps
! nprint : print-out frequency
! nsavc  : the trajectory saving frequency
!

! estimate Pmass from SYSmass (total system mass)
! [there could be problems with exreme values, such as  Pmass << SYSmass or Pmass >> SYSmass
scalar mass stat
calc Pmass = int ( ?stot  /  50.0 )

set nstep = {nstep}
set temp = 303.15

!shak bonh para fast sele segi WAT end
shak bonh para fast sele segi SOLV end

set pcnt = 1
if pcnt .eq. 0 open read  unit 11 card name charmm_lig_in_waterbox.rst 
open write unit 13 file name charmm_lig_in_waterbox.dcd 

DYNA CPT leap restart time 0.002 nstep @nstep -
     nprint {steps_for_equilibration} iprfrq {steps_for_equilibration} ntrfrq {steps_for_equilibration} -
     iunread 11 iunwri 12 iuncrd 13 iunvel -1 kunit -1 -
     nsavc {nstdcd} nsavv 0 -
     PCONSTANT pref   1.0  pmass @Pmass  pgamma   20.0 -
     HOOVER    reft @temp  tmass 2000.0  tbath   @temp  firstt @temp
     echeck 0

stop"""

    if env == "vacuum":
        charmm_vacuum = f"{header}{gas_phase}"
        return charmm_vacuum
    elif env == "waterbox":
        charmm_waterbox = f"{header}{liquid_phase}"
        return charmm_waterbox