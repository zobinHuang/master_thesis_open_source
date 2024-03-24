# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import NvRules

def get_identifier():
    return "HighPipeUtilization"

def get_name():
    return "High Pipe Utilization"

def get_description():
    return "High pipe utilization bottleneck analysis"

def get_section_identifier():
    return "ComputeWorkloadAnalysis"

def apply(handle):
    ctx = NvRules.get_context(handle)
    action = ctx.range_by_idx(0).action_by_idx(0)
    fe = ctx.frontend()

    pipelines = {
        ("ADU",            70,  0, "sm__inst_executed_pipe_adu", None),
        ("ALU",            70,  0, "sm__pipe_alu_cycles_active", "executes integer and logic operations"),
        ("CBU",            70,  0, "sm__inst_executed_pipe_cbu", None),
        ("FMA",            70, 80, "sm__pipe_fma_cycles_active", "executes 32-bit floating point (FADD, FMUL, FMAD, ...) and integer (IMUL, IMAD) operations"),
        ("FP16",           70, 80, "sm__inst_executed_pipe_fp16", "executes 16-bit floating point operations"),
        ("FMA (FP16)",     86,  0, "sm__inst_executed_pipe_fma_type_fp16", "executes 16-bit floating point operations"),
        ("FP64",           70,  0, "sm__inst_executed_pipe_fp64", "executes 64-bit floating point operations"),
        ("LSU",            70,  0, "sm__inst_executed_pipe_lsu", "executes load/store memory operations"),
        ("Tensor (DP)",    80, 80, "sm__inst_executed_pipe_tensor_op_dmma", "executes 64-bit floating point tensor operations"),
        ("Tensor (FP)",    70,  0, "sm__inst_executed_pipe_tensor_op_hmma", "executes 16-bit floating point tensor operations"),
        ("Tensor (INT)",   72,  0, "sm__inst_executed_pipe_tensor_op_imma", "executes 4/8-bit integer tensor operations"),
        ("TEX",            70,  0, "sm__inst_executed_pipe_tex", "executes texture/surface operations"),
        ("Uniform",        75,  0, "sm__inst_executed_pipe_uniform", None),
        ("XU",             70,  0, "sm__inst_executed_pipe_xu", None),
    }

    metric_suffix = ".avg.pct_of_peak_sustained_active"

    low_utilization_threshold = 20
    high_utilization_threshold = 60
    bottleneck_utilization_threshold = 80
    max_utilization = 0.0
    max_pipe = None

    cc_major = action.metric_by_name("device__attribute_compute_capability_major").as_uint64()
    cc_minor = action.metric_by_name("device__attribute_compute_capability_minor").as_uint64()
    cc = cc_major * 10 + cc_minor

    for pipe in pipelines:
        if cc >= pipe[1] and (pipe[2] == 0 or cc <= pipe[2]):
            value = action.metric_by_name(pipe[3] + metric_suffix).as_double()
            if value > max_utilization:
                max_utilization = value
                max_pipe = pipe

    if max_pipe != None:
        if max_utilization < low_utilization_threshold:
            message = "All pipelines are under-utilized. Either this kernel is very small or it doesn't issue enough warps per scheduler."
            message += " Check the @section:LaunchStats:Launch Statistics@ and @section:SchedulerStats:Scheduler Statistics@ sections for further details."
            fe.message(NvRules.IFrontend.MsgType_MSG_WARNING, message)
        elif max_utilization < high_utilization_threshold:
            fe.message("No pipeline is over-utilized.")
        else:
            message = max_pipe[0] + " is the highest-utilized pipeline ({:.1f}%).".format(max_utilization)
            pipe_info = max_pipe[4]
            if pipe_info != None:
                message += " It " + pipe_info + "."

            if max_utilization < bottleneck_utilization_threshold:
                message += " The pipeline is well-utilized and might become a bottleneck if more work is added."
                fe.message(message)
            else:
                message += " The pipeline is over-utilized and likely a performance bottleneck."
                fe.message(NvRules.IFrontend.MsgType_MSG_WARNING, message)

