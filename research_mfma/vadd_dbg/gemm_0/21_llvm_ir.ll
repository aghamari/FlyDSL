; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"

define amdgpu_kernel void @gemm_0(ptr addrspace(1) %0, <{ <{ i32, i32 }>, <{ i64 }> }> %1, ptr addrspace(1) %2, <{ <{ i32, i32 }>, <{ i64 }> }> %3, ptr addrspace(1) %4, <{ <{ i32, i32 }>, <{ i64 }> }> %5) #0 !dbg !3 !reqd_work_group_size !7 {
  %7 = extractvalue <{ <{ i32, i32 }>, <{ i64 }> }> %5, 1, 0, !dbg !8
  %8 = extractvalue <{ <{ i32, i32 }>, <{ i64 }> }> %3, 1, 0, !dbg !8
  %9 = extractvalue <{ <{ i32, i32 }>, <{ i64 }> }> %1, 1, 0, !dbg !8
  %10 = call range(i32 0, 256) i32 @llvm.amdgcn.workitem.id.x(), !dbg !9
  %11 = sext i32 %10 to i64, !dbg !9
  %12 = trunc i64 %11 to i32, !dbg !9
  %13 = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1(ptr addrspace(1) %0, i16 0, i64 4294967295, i32 159744), !dbg !12
  %14 = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1(ptr addrspace(1) %2, i16 0, i64 4294967295, i32 159744), !dbg !13
  %15 = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1(ptr addrspace(1) %4, i16 0, i64 4294967295, i32 159744), !dbg !14
  %16 = mul i64 %9, 16, !dbg !15
  %17 = srem i32 %12, 16, !dbg !15
  %18 = sdiv i32 %12, 16, !dbg !15
  %19 = sext i32 %17 to i64, !dbg !15
  %20 = mul i64 %19, %9, !dbg !15
  %21 = srem i32 %18, 4, !dbg !15
  %22 = sdiv i32 %18, 4, !dbg !15
  %23 = mul i32 %21, 4, !dbg !15
  %24 = sext i32 %23 to i64, !dbg !15
  %25 = add i64 %20, %24, !dbg !15
  %26 = srem i32 %22, 2, !dbg !15
  %27 = sext i32 %26 to i64, !dbg !15
  %28 = mul i64 %27, %16, !dbg !15
  %29 = add i64 %25, %28, !dbg !15
  %30 = mul i64 %8, 16, !dbg !16
  %31 = mul i64 %19, %8, !dbg !16
  %32 = add i64 %31, %24, !dbg !16
  %33 = sdiv i32 %22, 2, !dbg !16
  %34 = sext i32 %33 to i64, !dbg !16
  %35 = mul i64 %34, %30, !dbg !16
  %36 = add i64 %32, %35, !dbg !16
  %37 = trunc i64 %29 to i32, !dbg !11
  %38 = mul i32 %37, 2, !dbg !17
  %39 = call i64 @llvm.amdgcn.raw.ptr.buffer.load.i64(ptr addrspace(8) %13, i32 %38, i32 0, i32 0), !dbg !17
  %40 = trunc i64 %36 to i32, !dbg !11
  %41 = mul i32 %40, 2, !dbg !18
  %42 = call i64 @llvm.amdgcn.raw.ptr.buffer.load.i64(ptr addrspace(8) %14, i32 %41, i32 0, i32 0), !dbg !18
  %43 = bitcast i64 %39 to <4 x i16>, !dbg !19
  %44 = bitcast i64 %42 to <4 x i16>, !dbg !19
  %45 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %43, <4 x i16> %44, <4 x float> zeroinitializer, i32 0, i32 0, i32 0), !dbg !19
  %46 = add i64 %29, 16, !dbg !11
  %47 = trunc i64 %46 to i32, !dbg !11
  %48 = mul i32 %47, 2, !dbg !17
  %49 = call i64 @llvm.amdgcn.raw.ptr.buffer.load.i64(ptr addrspace(8) %13, i32 %48, i32 0, i32 0), !dbg !17
  %50 = add i64 %36, 16, !dbg !11
  %51 = trunc i64 %50 to i32, !dbg !11
  %52 = mul i32 %51, 2, !dbg !18
  %53 = call i64 @llvm.amdgcn.raw.ptr.buffer.load.i64(ptr addrspace(8) %14, i32 %52, i32 0, i32 0), !dbg !18
  %54 = bitcast i64 %49 to <4 x i16>, !dbg !19
  %55 = bitcast i64 %53 to <4 x i16>, !dbg !19
  %56 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %54, <4 x i16> %55, <4 x float> %45, i32 0, i32 0, i32 0), !dbg !19
  %57 = add i64 %29, 32, !dbg !11
  %58 = trunc i64 %57 to i32, !dbg !11
  %59 = mul i32 %58, 2, !dbg !17
  %60 = call i64 @llvm.amdgcn.raw.ptr.buffer.load.i64(ptr addrspace(8) %13, i32 %59, i32 0, i32 0), !dbg !17
  %61 = add i64 %36, 32, !dbg !11
  %62 = trunc i64 %61 to i32, !dbg !11
  %63 = mul i32 %62, 2, !dbg !18
  %64 = call i64 @llvm.amdgcn.raw.ptr.buffer.load.i64(ptr addrspace(8) %14, i32 %63, i32 0, i32 0), !dbg !18
  %65 = bitcast i64 %60 to <4 x i16>, !dbg !19
  %66 = bitcast i64 %64 to <4 x i16>, !dbg !19
  %67 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %65, <4 x i16> %66, <4 x float> %56, i32 0, i32 0, i32 0), !dbg !19
  %68 = add i64 %29, 48, !dbg !11
  %69 = trunc i64 %68 to i32, !dbg !11
  %70 = mul i32 %69, 2, !dbg !17
  %71 = call i64 @llvm.amdgcn.raw.ptr.buffer.load.i64(ptr addrspace(8) %13, i32 %70, i32 0, i32 0), !dbg !17
  %72 = add i64 %36, 48, !dbg !11
  %73 = trunc i64 %72 to i32, !dbg !11
  %74 = mul i32 %73, 2, !dbg !18
  %75 = call i64 @llvm.amdgcn.raw.ptr.buffer.load.i64(ptr addrspace(8) %14, i32 %74, i32 0, i32 0), !dbg !18
  %76 = bitcast i64 %71 to <4 x i16>, !dbg !19
  %77 = bitcast i64 %75 to <4 x i16>, !dbg !19
  %78 = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16> %76, <4 x i16> %77, <4 x float> %67, i32 0, i32 0, i32 0), !dbg !19
  %79 = mul i64 %7, 4, !dbg !20
  %80 = srem i32 %18, 8, !dbg !20
  %81 = sdiv i32 %18, 8, !dbg !20
  %82 = sext i32 %80 to i64, !dbg !20
  %83 = mul i64 %82, %79, !dbg !20
  %84 = add i64 %19, %83, !dbg !20
  %85 = mul i32 %81, 16, !dbg !20
  %86 = sext i32 %85 to i64, !dbg !20
  %87 = add i64 %84, %86, !dbg !20
  %88 = trunc i64 %87 to i32, !dbg !11
  %89 = extractelement <4 x float> %78, i64 0, !dbg !21
  %90 = mul i32 %88, 4, !dbg !21
  %91 = bitcast float %89 to i32, !dbg !21
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %91, ptr addrspace(8) %15, i32 %90, i32 0, i32 0), !dbg !21
  %92 = add i64 %87, %7, !dbg !11
  %93 = trunc i64 %92 to i32, !dbg !11
  %94 = extractelement <4 x float> %78, i64 1, !dbg !21
  %95 = mul i32 %93, 4, !dbg !21
  %96 = bitcast float %94 to i32, !dbg !21
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %96, ptr addrspace(8) %15, i32 %95, i32 0, i32 0), !dbg !21
  %97 = mul i64 %7, 2, !dbg !21
  %98 = add i64 %87, %97, !dbg !11
  %99 = trunc i64 %98 to i32, !dbg !11
  %100 = extractelement <4 x float> %78, i64 2, !dbg !21
  %101 = mul i32 %99, 4, !dbg !21
  %102 = bitcast float %100 to i32, !dbg !21
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %102, ptr addrspace(8) %15, i32 %101, i32 0, i32 0), !dbg !21
  %103 = mul i64 %7, 3, !dbg !21
  %104 = add i64 %87, %103, !dbg !11
  %105 = trunc i64 %104 to i32, !dbg !11
  %106 = extractelement <4 x float> %78, i64 3, !dbg !21
  %107 = mul i32 %105, 4, !dbg !21
  %108 = bitcast float %106 to i32, !dbg !21
  call void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32 %108, ptr addrspace(8) %15, i32 %107, i32 0, i32 0), !dbg !21
  ret void, !dbg !22
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #1

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1(ptr addrspace(1) readnone, i16, i64, i32) #2

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: read)
declare i64 @llvm.amdgcn.raw.ptr.buffer.load.i64(ptr addrspace(8) readonly captures(none), i32, i32, i32 immarg) #3

; Function Attrs: convergent nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare <4 x float> @llvm.amdgcn.mfma.f32.16x16x16bf16.1k(<4 x i16>, <4 x i16>, <4 x float>, i32 immarg, i32 immarg, i32 immarg) #4

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: write)
declare void @llvm.amdgcn.raw.ptr.buffer.store.i32(i32, ptr addrspace(8) writeonly captures(none), i32, i32, i32 immarg) #5

attributes #0 = { "amdgpu-flat-work-group-size"="256,256" "uniform-work-group-size" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { nocallback nofree nosync nounwind willreturn memory(argmem: read) }
attributes #4 = { convergent nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none) }
attributes #5 = { nocallback nofree nosync nounwind willreturn memory(argmem: write) }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "MLIR", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly)
!1 = !DIFile(filename: "<unknown>", directory: "")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = distinct !DISubprogram(name: "gemm_0", linkageName: "gemm_0", scope: !4, file: !4, line: 122, type: !5, scopeLine: 122, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!4 = !DIFile(filename: "gemm_2x2wave.py", directory: "/workspaces/workspace/FlyDSL/research_mfma")
!5 = !DISubroutineType(cc: DW_CC_normal, types: !6)
!6 = !{}
!7 = !{i32 256, i32 1, i32 1}
!8 = !DILocation(line: 114, scope: !3)
!9 = !DILocation(line: 70, column: 10, scope: !10, inlinedAt: !11)
!10 = distinct !DILexicalBlockFile(scope: !3, file: !4, discriminator: 0)
!11 = !DILocation(line: 122, column: 4, scope: !3)
!12 = !DILocation(line: 71, column: 8, scope: !10, inlinedAt: !11)
!13 = !DILocation(line: 72, column: 8, scope: !10, inlinedAt: !11)
!14 = !DILocation(line: 73, column: 8, scope: !10, inlinedAt: !11)
!15 = !DILocation(line: 94, column: 13, scope: !10, inlinedAt: !11)
!16 = !DILocation(line: 95, column: 13, scope: !10, inlinedAt: !11)
!17 = !DILocation(line: 103, column: 8, scope: !10, inlinedAt: !11)
!18 = !DILocation(line: 104, column: 8, scope: !10, inlinedAt: !11)
!19 = !DILocation(line: 105, column: 8, scope: !10, inlinedAt: !11)
!20 = !DILocation(line: 109, column: 38, scope: !10, inlinedAt: !11)
!21 = !DILocation(line: 109, column: 4, scope: !10, inlinedAt: !11)
!22 = !DILocation(line: 114, column: 4, scope: !10, inlinedAt: !11)
