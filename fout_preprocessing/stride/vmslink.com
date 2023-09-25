$ver = f$verify(1)
$cc/optimize stride,splitstr,rdpdb,initchn,geometry,thr2one
$cc/optimize one2thr,filename,tolostr,strutil,place_h,hbenergy
$cc/optimize memory,helix,sheet,rdmap,phipsi,command
$cc/optimize molscr,die,hydrbond,mergepat,fillasn,escape
$cc/optimize p_jrnl,p_rem,p_atom,p_helix,p_sheet,p_turn
$cc/optimize p_ssbond,p_expdta,p_model,p_compnd,report,nsc
$cc/optimize area,ssbond,chk_res,chk_atom,turn,pdbasn
$cc/optimize dssp,outseq,chkchain,elem,measure,asngener
$cc/optimize p_endmdl,stred
$
$
$link stride,splitstr,rdpdb,initchn,geometry,thr2one, -
one2thr,filename,tolostr,strutil,place_h,hbenergy, -
memory,helix,sheet,rdmap,phipsi,command, -
molscr,die,hydrbond,mergepat,fillasn,escape, -
p_jrnl,p_rem,p_atom,p_helix,p_sheet,p_turn, -
p_ssbond,p_expdta,p_model,p_compnd,report,nsc, -
area,ssbond,chk_res,chk_atom,turn,pdbasn, -
dssp,outseq,chkchain,elem,measure,asngener, -
p_endmdl,stred
$ver = f$verify(0)
