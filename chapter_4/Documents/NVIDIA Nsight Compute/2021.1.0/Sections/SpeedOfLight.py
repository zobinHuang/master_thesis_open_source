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
import math

def get_identifier():
    return "SOLBottleneck"

def get_name():
    return "Bottleneck"

def get_description():
    return "High-level bottleneck detection"

def get_section_identifier():
    return "SpeedOfLight"

def apply(handle):
    ctx = NvRules.get_context(handle)
    action = ctx.range_by_idx(0).action_by_idx(0)
    fe = ctx.frontend()

    num_waves = action.metric_by_name("launch__waves_per_multiprocessor").as_double()
    smSolPct = action.metric_by_name("sm__throughput.avg.pct_of_peak_sustained_elapsed").as_double()
    memSolPct = action.metric_by_name("gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed").as_double()

    balanced_threshold = 10
    latency_bound_threshold = 60
    no_bound_threshold = 80
    waves_threshold = 1

    msg_type = NvRules.IFrontend.MsgType_MSG_OK

    if smSolPct < no_bound_threshold and memSolPct < no_bound_threshold:
        if smSolPct < latency_bound_threshold and memSolPct < latency_bound_threshold:
            msg_type = NvRules.IFrontend.MsgType_MSG_WARNING
            if num_waves < waves_threshold:
                message = "This kernel grid is too small to fill the available resources on this device, resulting in only {:.1f} full waves across all SMs. Look at @section:LaunchStats:Launch Statistics@ for more details.".format(num_waves)
            else:
                message = "This kernel exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below {:.1f}% of peak typically indicate latency issues. Look at @section:SchedulerStats:Scheduler Statistics@ and @section:WarpStateStats:Warp State Statistics@ for potential reasons.".format(latency_bound_threshold)
        elif math.fabs(smSolPct - memSolPct) >= balanced_threshold:
            msg_type = NvRules.IFrontend.MsgType_MSG_WARNING
            if smSolPct > memSolPct:
                message = "Compute is more heavily utilized than Memory: Look at the @section:ComputeWorkloadAnalysis:Compute Workload Analysis@ report section to see what the compute pipelines are spending their time doing. Also, consider whether any computation is redundant and could be reduced or moved to look-up tables."
            else:
                message = "Memory is more heavily utilized than Compute: Look at the @section:MemoryWorkloadAnalysis:Memory Workload Analysis@ report section to see where the memory system bottleneck is. Check memory replay (coalescing) metrics to make sure you're efficiently utilizing the bytes transferred. Also consider whether it is possible to do more work per memory access (kernel fusion) or whether there are values you can (re)compute."
        else:
            message = "Compute and Memory are well-balanced: To reduce runtime, both computation and memory traffic must be reduced. Check both the @section:ComputeWorkloadAnalysis:Compute Workload Analysis@ and @section:MemoryWorkloadAnalysis:Memory Workload Analysis@ report sections."
    else:
        bottleneck_section = "@section:ComputeWorkloadAnalysis:Compute Workload Analysis@" if smSolPct >= memSolPct else "@section:MemoryWorkloadAnalysis:Memory Workload Analysis@"
        message = "The kernel is utilizing greater than {:.1f}% of the available compute or memory performance of the device. To further improve performance, work will likely need to be shifted from the most utilized to another unit. Start by analyzing workloads in the {} section.".format(no_bound_threshold, bottleneck_section)

    fe.message(msg_type, message)
