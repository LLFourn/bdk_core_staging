macro_rules! delegate_inner {
    (
        $(fn $fn_name:ident$(<$($tpl:ident  $(: $tcl:ident)?),*>)?(&self, $($arg:ident : $type:path),*) ->  $return:path);*
    ) => {

        $(
            fn $fn_name$(<$($tpl $(:$tcl)?),*>)?(&self, $($arg : $type)) -> $return {
                self.inner.$fn_name($($arg),*)
            }
        )*
    }
}
