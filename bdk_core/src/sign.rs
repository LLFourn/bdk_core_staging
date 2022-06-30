use bitcoin::secp256k1::{KeyPair, PublicKey, Secp256k1, SecretKey, Signing, Verification};
use bitcoin::util::bip32::{self, DerivationPath, ExtendedPrivKey};
use bitcoin::util::psbt::PartiallySignedTransaction as Psbt;
use bitcoin::util::sighash::SighashCache;
use bitcoin::util::taproot;
use bitcoin::{SchnorrSig, Transaction, XOnlyPublicKey};
use core::ops::Deref;
use miniscript::descriptor::{DescriptorSecretKey, DescriptorXKey, InnerXKey};
use miniscript::psbt::PsbtExt;

#[derive(Clone, Debug)]
pub enum SigningError {
    SigHashError(miniscript::psbt::SighashError),
    DerivationError(bip32::Error),
}

impl From<miniscript::psbt::SighashError> for SigningError {
    fn from(e: miniscript::psbt::SighashError) -> Self {
        Self::SigHashError(e)
    }
}

impl core::fmt::Display for SigningError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            SigningError::SigHashError(e) => e.fmt(f),
            SigningError::DerivationError(e) => e.fmt(f),
        }
    }
}

impl From<bip32::Error> for SigningError {
    fn from(e: bip32::Error) -> Self {
        Self::DerivationError(e)
    }
}

#[cfg(feature = "std")]
impl std::error::Error for SigningError {}

pub fn sign_with_single_key<T: Deref<Target = Transaction>>(
    secret_key: &SecretKey,
    psbt: &mut Psbt,
    sighash_cache: &mut SighashCache<T>,
    index: usize,
    secp: &Secp256k1<impl Signing + Verification>,
) -> Result<bool, SigningError> {
    let pubkey = PublicKey::from_secret_key(&secp, secret_key);
    let x_only_pubkey = XOnlyPublicKey::from(pubkey);
    let mut signed = false;

    if let Some(tap_internal_key) = psbt.inputs[index].tap_internal_key {
        if tap_internal_key == x_only_pubkey {
            let tweak = taproot::TapTweakHash::from_key_and_tweak(x_only_pubkey, None);
            let mut keypair = KeyPair::from_secret_key(&secp, secret_key.clone());
            keypair.tweak_add_assign(&secp, &tweak).unwrap();

            let msg = psbt.sighash_msg(index, sighash_cache, None)?;
            let sig = secp.sign_schnorr_no_aux_rand(&msg.to_secp_msg(), &keypair);
            let bitcoin_sig = SchnorrSig {
                sig,
                hash_ty: bitcoin::SchnorrSighashType::Default,
            };
            psbt.inputs[index].tap_key_sig = Some(bitcoin_sig);

            signed = true;
        }
    } else {
        todo!()
    }

    Ok(signed)
}

pub fn sign_with_xkey<T: Deref<Target = Transaction>>(
    xkey: &ExtendedPrivKey,
    origin: (bip32::Fingerprint, bip32::DerivationPath),
    psbt: &mut Psbt,
    sighash_cache: &mut SighashCache<T>,
    index: usize,
    secp: &Secp256k1<impl Signing + Verification>,
) -> Result<bool, SigningError> {
    let mut signed = false;
    let (my_fingerprint, my_path) = origin;
    if let Some(tap_internal_key) = psbt.inputs[index].tap_internal_key {
        if let Some((_, (fingerprint, path))) =
            psbt.inputs[index].tap_key_origins.get(&tap_internal_key)
        {
            if fingerprint == &my_fingerprint {
                if path[..].starts_with(&my_path[..]) {
                    let remaining: DerivationPath = path[my_path.len()..].into();
                    let derived_key = xkey.derive_priv(secp, &remaining)?;
                    signed = sign_with_single_key(
                        &derived_key.private_key,
                        psbt,
                        sighash_cache,
                        index,
                        secp,
                    )?
                }
            }
        }
    } else {
        todo!()
    }

    Ok(signed)
}

pub fn sign_with_descriptor_xkey<T: Deref<Target = Transaction>>(
    dxk: &DescriptorXKey<ExtendedPrivKey>,
    psbt: &mut Psbt,
    sighash_cache: &mut SighashCache<T>,
    index: usize,
    secp: &Secp256k1<impl Signing + Verification>,
) -> Result<bool, SigningError> {
    let xkey = &dxk.xkey;
    let origin = dxk
        .origin
        .clone()
        .unwrap_or_else(|| (xkey.xkey_fingerprint(secp), DerivationPath::master()));
    sign_with_xkey(&xkey, origin, psbt, sighash_cache, index, secp)
}

pub fn sign_with_descriptor_sk<T: Deref<Target = Transaction>>(
    dsk: &DescriptorSecretKey,
    psbt: &mut Psbt,
    sighash_cache: &mut SighashCache<T>,
    index: usize,
    secp: &Secp256k1<impl Signing + Verification>,
) -> Result<bool, SigningError> {
    match dsk {
        DescriptorSecretKey::Single(single_priv) => {
            sign_with_single_key(&single_priv.key.inner, psbt, sighash_cache, index, secp)
        }
        DescriptorSecretKey::XPrv(descriptor_xkey) => {
            sign_with_descriptor_xkey(descriptor_xkey, psbt, sighash_cache, index, secp)
        }
    }
}
