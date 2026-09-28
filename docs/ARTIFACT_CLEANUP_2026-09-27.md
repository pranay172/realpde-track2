# Artifact cleanup — 2026-09-27

Scope: Track 2 generated artifacts only; no datasets, environments, organizer
weights, submission code, or experiment controls removed.

## Outcome and verification

- Removed 59 files, totaling 20,991,154,620 bytes (**19.55 GiB**).
- Artifact tree apparent size: 31,374,269,138 → 10,383,114,518 bytes
  (**29.22 → 9.67 GiB**, approximately 67% smaller).
- All 48 retained files over 100 MiB have unchanged SHA-256 hashes after
  deletion, including every submission zip and extracted model.
- All 20 E031/E032 packed replacements loaded strictly in evaluator-compatible
  Torch 2.2.2 and matched both training and evaluation report hashes.
- Full unit suite: **108/108 passed** with the loader fix; `git diff --check`
  passed. No model recipe, deployment code, or predictions were changed.

## Retention and consumers

- E004/E009/E010/E011/E013/E014/E015: delete full-precision `final.pth`
  and intermediate `latest.pth`; retain each sibling `fno_fp16.pth` and all
  reports/logs. `train_fno.py --eval-only` prefers that packed checkpoint;
  downstream experiment configs use packed checkpoints, not these deleted files.
- E031/E032: delete 20 `final_fp32.pth` copies, retaining **all 20** sibling
  `fno_dual_head_fp16.pth` files, controls, training/evaluation reports, and logs.
  Both study runners' resume guards consume packed checkpoints and reports.
  The dual-head trainer's eval-only fallback now uses its tested format-aware
  loader (the previous code passed an unsupported `Path` to `unpack_fp16`).
  Re-evaluation from retained packed weights uses deployment precision, not the
  deleted pre-quantization fp32 state; historical fp32 results remain evidence.
- Delete only the listed smoke-test weights. Retain their reports/configs/logs;
  regenerate weights by rerunning the corresponding smoke command if needed.
- Delete nine E023–E026 replay arrays in `prelim_probe`; keep all JSON results,
  metadata, scripts, and logs. Historical consumers are `probe_e023.py`,
  `probe_e026.py`, and `evaluate_variance_head.py` (including archived copies).
  Before replay, regenerate with `scripts/probe_e023.py --phase collect-cal`,
  `collect-val`, and `collect-audit` using `../.venv-gpu/bin/python` (GPU access
  requires the usual approval). E020 packed weights and released data remain.
- Preserve every submission zip and extracted submission, including exact E029
  and E033 archives. Preserve every non-smoke packed checkpoint and the fp32
  E020/E021/E030/E033 anchors. No active training/evaluation process was found.

Deletion is permanent, not a move to trash. Full-precision originals and
intermediate checkpoints cannot be recovered losslessly from quantized weights;
retraining can regenerate them but bitwise identity is not guaranteed. Removed
replay arrays are derived only from released data and can be recollected.

## Exact removal manifest

Paths are relative to the repository root. Hashes were recorded before removal.

| Path | Bytes | SHA-256 |
|---|---:|---|
| `artifacts/e004/final.pth` | 402784381 | `dc559f9d15f1a9644b86db5f8dddb215fa411e0e7911e9ff2c886822c7290375` |
| `artifacts/e004/gpu_bs4/smoke/smoke_final.pth` | 402777613 | `02c87ad33657f1d25b69ef844fe38cee63584badd7e8c2aac506b03b6d1e63d8` |
| `artifacts/e004/gpu_bs4/smoke/smoke_fno_fp16.pth` | 201396405 | `0bb2731211378cb6878059b62dbedd5659ea2a63c99939b922696a77a9b50643` |
| `artifacts/e004/latest.pth` | 402776181 | `1a64d6489c91b7be651d0540d6d6e28d72c811b9551ef635fbc3b50dac475c5a` |
| `artifacts/e004/smoke/smoke_final.pth` | 402777200 | `ecf5d96b1d620753899bc5610796322645f1e7af654da6311cef524728500b4b` |
| `artifacts/e004/smoke/smoke_fno_fp16.pth` | 201395986 | `3605e67ae5ba951c7a622336d7d8bfc7b32ad754e55583dd8e66a0722a3a7939` |
| `artifacts/e009/final.pth` | 402795197 | `01ec9bf76cc43ddcf9b4a58140132538e144a1a6108083fe7d54eed9f4a2f354` |
| `artifacts/e009/latest.pth` | 402776181 | `b4cd496c216884731dff88fa13d13766c421946456c2828d75a1eb76506af955` |
| `artifacts/e010/final.pth` | 402784573 | `9f680acfc5a5c8f90b50b069472ad17fa36c2a6887caa8fbaae8b01d7ace0fb4` |
| `artifacts/e010/latest.pth` | 402776181 | `987019fd3e9306e14bf9ee5429e10cee9cae4421a0ed35527c8a222c97a2b183` |
| `artifacts/e011/final.pth` | 402784701 | `4d42a3b9953c89c940ffd1ea28e005720ae77d16e61cc2340f37bac7f8949c40` |
| `artifacts/e011/latest.pth` | 402776181 | `19887cc59397187095d4e7c805a5ff71f49691e5691bed80d4c66d2d26257fe4` |
| `artifacts/e011/smoke/smoke_final.pth` | 402777933 | `84c0b43809dbd0029c573c4ca5f3b1aabe8a77522826a1c8dbe6b577ebaaed3d` |
| `artifacts/e011/smoke/smoke_fno_fp16.pth` | 201396405 | `f5688b3c4935971891f5a6eb4764caf1defd61df5a54eb15dcebbb4b182df59f` |
| `artifacts/e013/final.pth` | 402784573 | `591e61a5a1cbe50cafa2ed3bc0dc061dc74bc18cc8ea429a9f046119b50ea74e` |
| `artifacts/e013/latest.pth` | 402776181 | `e40bb91764eb9d86348bd041e22b750314c60754a06bfd652125de5ee67bb6bc` |
| `artifacts/e013/smoke/smoke_final.pth` | 402777805 | `9c1beee3f1ae941276694e4b67183c28666b59f04c334d8f7524b48e0c103949` |
| `artifacts/e013/smoke/smoke_fno_fp16.pth` | 201396405 | `5d2daeb3be4034db3041e5bc61fe6f79db6f6582db1b02e0713f1a9147935cc6` |
| `artifacts/e014/final.pth` | 402784765 | `6a768388a410b52568addc1ef9f1ab09171557d9fdfe9f2ed0bae1b95ff912a2` |
| `artifacts/e014/latest.pth` | 402776181 | `f8e8160676196290ee0233094ac9159c1d109ad0d297b60730a2f182439bfaae` |
| `artifacts/e014/smoke/smoke_final.pth` | 402777997 | `7ad4f564f5bfa5383695a5712c0ce314b2e5e39d181b21231c88302f82fe552f` |
| `artifacts/e014/smoke/smoke_fno_fp16.pth` | 201396405 | `1c0b5d986d9dff61df96189aa2482864128836980ebe63339fd437622d37662f` |
| `artifacts/e015/final.pth` | 402785213 | `946e09a58012fbb7cd69d575e73f99cff2344b5fc60d8a33f73fb4702905ca30` |
| `artifacts/e015/latest.pth` | 402776181 | `63ef09744d7dce01a09e6bdba8f3a3475f9aefc7682ad467e151d3af03259d4e` |
| `artifacts/e015/smoke/smoke_final.pth` | 402777613 | `6376ad38cb7d5178cd6eb974ac6f2e0a2974cc0246e44e4f74700e29bbdc11e0` |
| `artifacts/e015/smoke/smoke_fno_fp16.pth` | 201396405 | `98c10b42549b390fe5617ef330b528ea8bab1dc5ec8df1cc88e1e2ab3ace3d7d` |
| `artifacts/e020/smoke/final_fp32.pth` | 402778397 | `fd939e0862df36b084cea02bc486d7f0527fd2bcd4c7c5a2526ede528b0246f1` |
| `artifacts/e020/smoke/fno_dual_head_fp16.pth` | 201398125 | `b758b134e3bfc3eeb4ef3c2e70df8836edf87bab78d8e3491eab842944a52692` |
| `artifacts/e030/smoke/final_fp32.pth` | 402777986 | `ee2371fed22410ad5e1ddaa266a2fb8cecdef8f1667f4091ac4947a40237f1ad` |
| `artifacts/e030/smoke/fno_dual_head_fp16.pth` | 201397698 | `1ae1638b9240a270558a7639c84063200718b3d25566851a32685b39e6beb57a` |
| `artifacts/e031/real_regime_complement_v1_checkpoint_s0/final_fp32.pth` | 402778397 | `c469c24e0fd504a8b29fb6137d482dc410456db5e5ad30ffa3429b85f71acfa3` |
| `artifacts/e031/real_regime_complement_v1_checkpoint_s1/final_fp32.pth` | 402778397 | `d3c8622d89d97c1db5bb62038cb44577f2ef74da69f9a99868f2be369c35f3d5` |
| `artifacts/e031/real_regime_complement_v1_random_s0/final_fp32.pth` | 402778397 | `833ad7a070e1dc4598ad259a11ff85c227fd2566aa7060f4ca848ad51bd4917d` |
| `artifacts/e031/real_regime_complement_v1_random_s1/final_fp32.pth` | 402778397 | `264b02564561fc89be32a514183c51e105eddbe983101b713c65069263d70417` |
| `artifacts/e031/real_regime_v1_checkpoint_s0/final_fp32.pth` | 402778397 | `8d9053be6413c405640b0ea827f85d1c2ddba69d439a74dbeba3c4cac5012cbc` |
| `artifacts/e031/real_regime_v1_checkpoint_s1/final_fp32.pth` | 402778397 | `a0b5ffa5b02d3f99b1a1a5cc6638ef89def6f9587bbedead5f02d005768b6607` |
| `artifacts/e031/real_regime_v1_random_s0/final_fp32.pth` | 402778397 | `03454a439a27a55476048b923f69886111e306af2c199a5bfcfde9275b6c1a9b` |
| `artifacts/e031/real_regime_v1_random_s1/final_fp32.pth` | 402778397 | `9318d2e9188a1d79837063bcfca6643b220cb7669fe70eca6db4fc0b1031fc22` |
| `artifacts/e032/real_regime_complement_v1_stride1_u1800_checkpoint_s0/final_fp32.pth` | 402778397 | `d118a6ad0b0d9b8fd8aae5abfcf569e0fb0a752f14e5a1964d7392a3388546d7` |
| `artifacts/e032/real_regime_complement_v1_stride1_u600_checkpoint_s0/final_fp32.pth` | 402778397 | `86a1bae8f08fe2798ca45237e181d0fae0d2a7d023da32861c11a49e1bbc2559` |
| `artifacts/e032/real_regime_complement_v1_stride20_u1800_checkpoint_s0/final_fp32.pth` | 402778397 | `f909941ea9486fd10a9d09d09d4d9e7dea50b29b45bd7ed8bdff784511a30d24` |
| `artifacts/e032/real_regime_complement_v1_stride20_u1800_checkpoint_s1/final_fp32.pth` | 402778397 | `d6dc5ad16352cb9508e7aa9e2318759ca45b371d3ef5c1c84c0e68dc59ca5d05` |
| `artifacts/e032/real_regime_complement_v1_stride20_u600_checkpoint_s0/final_fp32.pth` | 402778397 | `aa407c847e553030ad34ee6205f799ef3905df9cce9e120fb474a9dcca6ec457` |
| `artifacts/e032/real_regime_complement_v1_stride20_u600_checkpoint_s1/final_fp32.pth` | 402778397 | `105a7ddb091f5cfe26dfdeb37a3928999668d85bfd4304d163e994f2f32c988a` |
| `artifacts/e032/real_regime_v1_stride1_u1800_checkpoint_s0/final_fp32.pth` | 402778397 | `4932948c3a466a66ea588555e8db29ed9d83e3ab3264f6faf59cfc969132a36b` |
| `artifacts/e032/real_regime_v1_stride1_u600_checkpoint_s0/final_fp32.pth` | 402778397 | `74ad0468e93ba9ee4325b3475cdf14ecfc066d694638fcb9aca2bdfb53c7a810` |
| `artifacts/e032/real_regime_v1_stride20_u1800_checkpoint_s0/final_fp32.pth` | 402778397 | `9a7e8f10f313495b9bd416eb734da3ba1a0696a7b2d09384bf95ae4083ec6822` |
| `artifacts/e032/real_regime_v1_stride20_u1800_checkpoint_s1/final_fp32.pth` | 402778397 | `2f1709da52eb2190a9db27a901b6792a94521ce19c05e127b0ff7a5e739d826e` |
| `artifacts/e032/real_regime_v1_stride20_u600_checkpoint_s0/final_fp32.pth` | 402778397 | `55e9cec5b281942c49c7628daa65d3ca8dad0aadc03dcb334f02fb30a2fbfb22` |
| `artifacts/e032/real_regime_v1_stride20_u600_checkpoint_s1/final_fp32.pth` | 402778397 | `03b10397db33313c402793dc1339af39697ebd07cb4d908af5ad89c936491f45` |
| `artifacts/prelim_probe/audit_raw.npy` | 420003968 | `b28882b1ef12cd3796b31a3180edab01c6289a760ab5d674a9017f7639b1c564` |
| `artifacts/prelim_probe/audit_x.npy` | 420003968 | `fa562786a1a0c48c884984e07ecb4bb7ec171d72b5c8ca13be5a8353a065b49b` |
| `artifacts/prelim_probe/audit_y.npy` | 420003968 | `b347772c9574f2e572bd36638dfd2fee6758b5a1edbff0a9f0897f3ac0da214b` |
| `artifacts/prelim_probe/cal_raw.npy` | 147701888 | `059a2bd6e9f1198438580571f9d0ec83fe7a8ab2aa6cb842bfd2150fee8dc845` |
| `artifacts/prelim_probe/cal_x.npy` | 147701888 | `45a631a69b9167fe94237f96bd2014d76b82bd0966cfe1d7e12beee8b804da2d` |
| `artifacts/prelim_probe/cal_y.npy` | 147701888 | `b33ee4e368cbc6b1f45cb3bd908dd6aa4ff86df2de6ae81737d2e3e0a9071dab` |
| `artifacts/prelim_probe/val_raw.npy` | 253378688 | `d00e024412e9d16ff1295d59d9f6ea8ec468961b2560523c5180981e132f8140` |
| `artifacts/prelim_probe/val_x.npy` | 253378688 | `296f8758c221153ebba27c85850f2a08c68ea088fbd820d285dfd6014497b7bb` |
| `artifacts/prelim_probe/val_y.npy` | 253378688 | `76246836c81c374582229efd2168f52dbb1a77e44900fcf8700f841cadcfdadd` |

## Retained large-file integrity inventory

All files over 100 MiB not selected above; hashes captured before cleanup for
post-cleanup verification. Small reports/configurations/logs are retained too.

| Path | Bytes | SHA-256 |
|---|---:|---|
| `artifacts/e004/fno_fp16.pth` | 201396069 | `75cc07b42bfeb74f2a204dde26650a7f07289c377e35bfe04356f5bbfb9d363c` |
| `artifacts/e007/e007_candidate.zip` | 186181793 | `2f2f3dcad4a43618ea5c72836cee71007f8602846943a2e24d1b6ca5c12b4ba5` |
| `artifacts/e007/extracted/model.pth` | 201396069 | `75cc07b42bfeb74f2a204dde26650a7f07289c377e35bfe04356f5bbfb9d363c` |
| `artifacts/e009/fno_fp16.pth` | 201396069 | `b821e85d28626ee43da6e5ba981278edb2d289267900072f5a8c754344cf4a4b` |
| `artifacts/e010/fno_fp16.pth` | 201396069 | `89b2a522f717645f5f03cd3c011c76c810252304f422187936d00006ae9bc2d5` |
| `artifacts/e011/fno_fp16.pth` | 201396069 | `df54ea7b90b48530f2b186f5becf3b5215c96b23ac91f9e2137a21b174d1f4b5` |
| `artifacts/e013/fno_fp16.pth` | 201396069 | `a2001324e4b9ee1cd49b90feb0e65a27d9c737c20429528cabcf315f98b99334` |
| `artifacts/e014/fno_fp16.pth` | 201396069 | `aadf61e7d049ae77f86a71878cdf63acc3c8cca8b9db8d6edc7e4df4630c1979` |
| `artifacts/e015/fno_fp16.pth` | 201396069 | `1e61d570bf941985d5caa0a033ada7c59a1d609c38eef918ae0058583ba73343` |
| `artifacts/e016/e016_candidate.zip` | 186177568 | `6c55286aa7d2b08d6238b8cb6040efc3adf94a6fb93c9ac944b2729fbc258f86` |
| `artifacts/e016/extracted/model.pth` | 201396069 | `1e61d570bf941985d5caa0a033ada7c59a1d609c38eef918ae0058583ba73343` |
| `artifacts/e020/final_fp32.pth` | 402778165 | `d80dd8937ec2f4f8bc3c559722f853eb450c3c7ebcd31e6d22fff64f91335b3a` |
| `artifacts/e020/fno_dual_head_fp16.pth` | 201397829 | `b371a2ce326595b62aa6fda8c75e2679dfdc409eb4acef618a1a628810db7534` |
| `artifacts/e021/final_fp32.pth` | 402778397 | `79eeb55349f5bc09c1efbdbc46a1f03155807e71bf2fc045814d80e75dcc4a7d` |
| `artifacts/e021/fno_dual_head_fp16.pth` | 201398125 | `3d9a6b93b8634db08b303d220a6a5df48ef35beb84628d8456d8fddc6d903dfe` |
| `artifacts/e022/e022_candidate.zip` | 185747618 | `38f83f1c4e7910ebe9a931a97ff9b9963f837757148bd595e730aae11a0d293e` |
| `artifacts/e022/extracted/model.pth` | 201398125 | `3d9a6b93b8634db08b303d220a6a5df48ef35beb84628d8456d8fddc6d903dfe` |
| `artifacts/e024/e024_candidate.zip` | 185748360 | `708b9a743ddf62a6c4e35617ed5c23b91906bb7f473ea91b13fff3fd8999f67f` |
| `artifacts/e024/extracted/model.pth` | 201398125 | `3d9a6b93b8634db08b303d220a6a5df48ef35beb84628d8456d8fddc6d903dfe` |
| `artifacts/e027/fno_dual_head_fp16.pth` | 201397829 | `1850189484f832ca4622f970cd50dd8c5b7ac9409a92dbdcfe950b9c2146adba` |
| `artifacts/e029/e029_candidate.zip` | 185748575 | `bdf06b5af726b99b97fe333fb55f3f5968c5cff3c0e890d5d3cbca7525ca08c4` |
| `artifacts/e029/extracted/model.pth` | 201398125 | `3d9a6b93b8634db08b303d220a6a5df48ef35beb84628d8456d8fddc6d903dfe` |
| `artifacts/e030/final_fp32.pth` | 402778397 | `c469c24e0fd504a8b29fb6137d482dc410456db5e5ad30ffa3429b85f71acfa3` |
| `artifacts/e030/fno_dual_head_fp16.pth` | 201398125 | `abe3c93ee504aaa467b1fd9969715df506ed139fc28b2f814b34e4029d81a333` |
| `artifacts/e031/real_regime_complement_v1_checkpoint_s0/fno_dual_head_fp16.pth` | 201398125 | `abe3c93ee504aaa467b1fd9969715df506ed139fc28b2f814b34e4029d81a333` |
| `artifacts/e031/real_regime_complement_v1_checkpoint_s1/fno_dual_head_fp16.pth` | 201398125 | `cca0f49b260dd9d9599a4d6b18c5b60f4a040793f15d950f9d4f1ba2b1e308ad` |
| `artifacts/e031/real_regime_complement_v1_random_s0/fno_dual_head_fp16.pth` | 201398125 | `e76ba490b42aff5deaa30db3ab78dbe15f3793a3419e4c80ef8989593d583e71` |
| `artifacts/e031/real_regime_complement_v1_random_s1/fno_dual_head_fp16.pth` | 201398125 | `81c9e36472c5f39feae877647cbf715bb65ebd5ee86d08e84b29003a4d6735a0` |
| `artifacts/e031/real_regime_v1_checkpoint_s0/fno_dual_head_fp16.pth` | 201398125 | `87a31c959d4eb080cd4458890fdfb6efff86d8f2ab29a35a2dc00d633a50174c` |
| `artifacts/e031/real_regime_v1_checkpoint_s1/fno_dual_head_fp16.pth` | 201398125 | `43216e6d6a6be369dc19445fc06e63fa8748413bf841ded5f3a1654b8d1e678d` |
| `artifacts/e031/real_regime_v1_random_s0/fno_dual_head_fp16.pth` | 201398125 | `4c064ea2edea6c0b26875c087aaa0f31b97d251e84872211f5cef09a1dc3fa36` |
| `artifacts/e031/real_regime_v1_random_s1/fno_dual_head_fp16.pth` | 201398125 | `16d8e692101004fdfa5b3ec2845029161f3c8c6f0c95a39e80ee0051f0edc8dd` |
| `artifacts/e032/real_regime_complement_v1_stride1_u1800_checkpoint_s0/fno_dual_head_fp16.pth` | 201398125 | `85c0de4dba124bed1d7f492da1744118ba9c912fa2d3850e5ceab6bad3ccdaea` |
| `artifacts/e032/real_regime_complement_v1_stride1_u600_checkpoint_s0/fno_dual_head_fp16.pth` | 201398125 | `ac981d423223508ae602192ada48dd971bc5db074719e4fdddab9ead4284eed3` |
| `artifacts/e032/real_regime_complement_v1_stride20_u1800_checkpoint_s0/fno_dual_head_fp16.pth` | 201398125 | `d452b2db3f2d5bd0cdffc84c3709345f52f4ba72765837f6562cbf1289150909` |
| `artifacts/e032/real_regime_complement_v1_stride20_u1800_checkpoint_s1/fno_dual_head_fp16.pth` | 201398125 | `5b70187d2be0247fa4e2669ca4ddbf6a6e230922688003b5282cb7b847991137` |
| `artifacts/e032/real_regime_complement_v1_stride20_u600_checkpoint_s0/fno_dual_head_fp16.pth` | 201398125 | `da3395dd96641e6b597fc6c5fd13320908a50296ad93bc4a97cecd9c57cd531d` |
| `artifacts/e032/real_regime_complement_v1_stride20_u600_checkpoint_s1/fno_dual_head_fp16.pth` | 201398125 | `c1d670c69a8a1ab2de46556f4cc0dfd0452b8ec6b8e4ad8c756966fc2d40749d` |
| `artifacts/e032/real_regime_v1_stride1_u1800_checkpoint_s0/fno_dual_head_fp16.pth` | 201398125 | `7842bbc102e2d76dc69bcae0d902ad9e8a4fd61c30b26171caf1f92baa76c1e1` |
| `artifacts/e032/real_regime_v1_stride1_u600_checkpoint_s0/fno_dual_head_fp16.pth` | 201398125 | `ee3c5bfc384df56686b0b6c2e78ca88b7a3a40933b21ab141df9ceba539a1ca8` |
| `artifacts/e032/real_regime_v1_stride20_u1800_checkpoint_s0/fno_dual_head_fp16.pth` | 201398125 | `b6c9a4cfef7960e1827f48c180b8e32bd55c9b1c1a4908ccc3decf64e0725318` |
| `artifacts/e032/real_regime_v1_stride20_u1800_checkpoint_s1/fno_dual_head_fp16.pth` | 201398125 | `c28d8f7b171fa40765a92daaac126f2b7c49739bc07bafbb687fa5a9721aff19` |
| `artifacts/e032/real_regime_v1_stride20_u600_checkpoint_s0/fno_dual_head_fp16.pth` | 201398125 | `93c6931baaf7b67c9a779fe528969d84738ce42886b1d53311e641ae0592fd65` |
| `artifacts/e032/real_regime_v1_stride20_u600_checkpoint_s1/fno_dual_head_fp16.pth` | 201398125 | `28b5f7fdb3a3a7fb2d5e16a4984cc7e921a46a35387af03413a92d40a2823802` |
| `artifacts/e033/e033_candidate.zip` | 186147375 | `50a5b3dc2550b4129eabfe917085e98aedceb0224188c21e269856ae60b6ba93` |
| `artifacts/e033/extracted/model.pth` | 201398125 | `2e7431c3e24587d5ad94ef6335908cbfb5ba034f86bccb0ec9566a67ddd81080` |
| `artifacts/e033/final_fp32.pth` | 402778397 | `476ead4671317b85ff3ec20b776da03bff1253cba2de76de9825edb2eab1aeb8` |
| `artifacts/e033/fno_dual_head_fp16.pth` | 201398125 | `2e7431c3e24587d5ad94ef6335908cbfb5ba034f86bccb0ec9566a67ddd81080` |
