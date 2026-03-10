#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/stl_bind.h>
#include "dft_engine.h"

namespace py = pybind11;
using namespace ase;

// helper: Chroma (std::array<double,12>) → numpy
static py::array_t<double> chroma_to_np(const Chroma& c) {
    py::array_t<double> a(12);
    std::memcpy(a.mutable_data(), c.data(), 12 * sizeof(double));
    return a;
}

PYBIND11_MODULE(ase_core, m) {
    m.doc() = "Automated Sound Engineer – C++ DFT core";

    // ---------- enums ----------
    py::enum_<Window>(m, "Window")
        .value("HANN",           Window::HANN)
        .value("HAMMING",        Window::HAMMING)
        .value("BLACKMAN_HARRIS",Window::BLACKMAN_HARRIS)
        .value("RECTANGULAR",    Window::RECTANGULAR);

    // ---------- structs ----------
    py::class_<Peak>(m, "Peak")
        .def(py::init<>())
        .def_readwrite("freq",  &Peak::freq)
        .def_readwrite("amp",   &Peak::amp)
        .def_readwrite("phase", &Peak::phase)
        .def_readwrite("bin",   &Peak::bin);

    py::class_<Partial>(m, "Partial")
        .def(py::init<>())
        .def_readwrite("id",     &Partial::id)
        .def_readwrite("times",  &Partial::times)
        .def_readwrite("freqs",  &Partial::freqs)
        .def_readwrite("amps",   &Partial::amps)
        .def_readwrite("phases", &Partial::phases)
        .def_readwrite("active", &Partial::active);

    py::class_<Note>(m, "Note")
        .def(py::init<>())
        .def_readwrite("start", &Note::start)
        .def_readwrite("end",   &Note::end)
        .def_readwrite("freq",  &Note::freq)
        .def_readwrite("amp",   &Note::amp)
        .def_readwrite("midi",  &Note::midi);

    // ---------- free functions ----------
    m.def("make_window", &make_window, py::arg("n"), py::arg("type"));
    m.def("compute_rms", [](py::array_t<double> buf) {
        auto r = buf.unchecked<1>();
        return compute_rms(r.data(0), r.size());
    }, py::arg("buffer"));
    m.def("freq_to_midi", &freq_to_midi);
    m.def("midi_to_freq", &midi_to_freq);
    m.def("midi_to_name", &midi_to_name);

    // raw FFT access
    m.def("fft", [](py::array_t<double> signal, int fft_size) {
        auto r = signal.unchecked<1>();
        CVec spec = fft_real(r.data(0), r.size(), fft_size);
        // return as Nx2 numpy  (real, imag)
        size_t N = spec.size();
        py::array_t<double> out({N, (size_t)2});
        auto o = out.mutable_unchecked<2>();
        for (size_t i = 0; i < N; ++i) {
            o(i, 0) = spec[i].real();
            o(i, 1) = spec[i].imag();
        }
        return out;
    }, py::arg("signal"), py::arg("fft_size") = 0);

    m.def("chroma_sequence_match", [](py::array_t<double> tpl, py::array_t<double> src) {
        py::buffer_info tpl_info = tpl.request();
        py::buffer_info src_info = src.request();
        
        if (tpl_info.ndim != 2 || src_info.ndim != 2)
            throw std::runtime_error("Inputs must be 2D arrays");
            
        int T = static_cast<int>(tpl_info.shape[0]);
        int S = static_cast<int>(src_info.shape[0]);
        int C = static_cast<int>(tpl_info.shape[1]);
        
        if (C != 12 || src_info.shape[1] != 12)
            throw std::runtime_error("Chroma arrays must have 12 columns");

        std::vector<double> result = ase::chroma_sequence_match(
            static_cast<const double*>(tpl_info.ptr), T,
            static_cast<const double*>(src_info.ptr), S, C);
            
        return py::array_t<double>(result.size(), result.data());
    }, "Cross-correlate a live chroma sequence against a reference window.");

    // ---------- STFT ----------
    py::class_<STFT::Frame>(m, "STFTFrame")
        .def_readwrite("time",      &STFT::Frame::time)
        .def_property_readonly("magnitude", [](const STFT::Frame& f){
            return py::array_t<double>(f.magnitude.size(),
                                       f.magnitude.data());
        })
        .def_property_readonly("mag_db", [](const STFT::Frame& f){
            return py::array_t<double>(f.mag_db.size(), f.mag_db.data());
        })
        .def_property_readonly("phase", [](const STFT::Frame& f){
            return py::array_t<double>(f.phase.size(), f.phase.data());
        });

    py::class_<STFT>(m, "STFT")
        .def(py::init<int, int, int, Window>(),
             py::arg("fft_size"), py::arg("hop_size"),
             py::arg("sample_rate"),
             py::arg("window") = Window::HANN)
        .def("analyze", &STFT::analyze, py::arg("signal"))
        .def("analyze_frame",
             [](const STFT& s, py::array_t<double> buf, double t) {
                 auto r = buf.unchecked<1>();
                 return s.analyze_frame(r.data(0), (int)r.size(), t);
             }, py::arg("samples"), py::arg("time"))
        .def("detect_peaks", &STFT::detect_peaks,
             py::arg("frame"), py::arg("threshold_db") = -60.0)
        .def("synthesize", &STFT::synthesize, py::arg("frames"))
        .def_property_readonly("fft_size",    &STFT::fft_size)
        .def_property_readonly("hop_size",    &STFT::hop_size)
        .def_property_readonly("sample_rate", &STFT::sample_rate)
        .def_property_readonly("freq_res",    &STFT::freq_res);

    // ---------- PartialTracker ----------
    py::class_<PartialTracker>(m, "PartialTracker")
        .def(py::init<double, double, int>(),
             py::arg("tolerance_cents") = 50.0,
             py::arg("min_partial_dur") = 0.03,
             py::arg("max_gap_frames")  = 3)
        .def("feed",      &PartialTracker::feed,
             py::arg("peaks"), py::arg("time"))
        .def("finish",    &PartialTracker::finish)
        .def("completed", &PartialTracker::completed)
        .def("all",       &PartialTracker::all)
        .def_static("extract_notes", &PartialTracker::extract_notes,
             py::arg("partials"),
             py::arg("min_note_dur")     = 0.05,
             py::arg("pitch_gate_cents") = 80.0);

    // ---------- ChromaExtractor ----------
    py::class_<ChromaExtractor>(m, "ChromaExtractor")
        .def(py::init<int, int, int, double>(),
             py::arg("fft_size"), py::arg("hop_size"),
             py::arg("sample_rate"), py::arg("tuning_ref") = 440.0)
        .def("analyze", [](const ChromaExtractor& ce, py::array_t<double> sig){
            auto r = sig.unchecked<1>();
            RVec v(r.data(0), r.data(0) + r.size());
            auto chromas = ce.analyze(v);
            // return as (N, 12) numpy
            size_t N = chromas.size();
            py::array_t<double> out({N, (size_t)12});
            auto o = out.mutable_unchecked<2>();
            for (size_t i = 0; i < N; ++i)
                for (int j = 0; j < 12; ++j)
                    o(i, j) = chromas[i][j];
            return out;
        }, py::arg("signal"))
        .def("analyze_frame", [](const ChromaExtractor& ce,
                                  py::array_t<double> mag){
            auto r = mag.unchecked<1>();
            RVec v(r.data(0), r.data(0) + r.size());
            Chroma c = ce.analyze_frame(v);
            return chroma_to_np(c);
        }, py::arg("magnitude"));

        // ---------- HPSS ----------
    py::class_<HPSS>(m, "HPSS")
        .def(py::init<int, int, int, int, int, double, Window>(),
             py::arg("fft_size"), py::arg("hop_size"), py::arg("sr"),
             py::arg("time_kernel")  = 17,
             py::arg("freq_kernel")  = 17,
             py::arg("mask_power")   = 2.0,
             py::arg("window")       = Window::HANN)
        .def("feed", [](HPSS& self, py::array_t<double> buf, double t) {
            auto r = buf.unchecked<1>();
            return self.feed(r.data(0), (int)r.size(), t);
        }, py::arg("samples"), py::arg("time"))
        .def("harmonic_magnitude", [](const HPSS& self) {
            auto& v = self.harmonic_magnitude();
            return py::array_t<double>(v.size(), v.data());
        })
        .def("harmonic_spectrum", [](const HPSS& self) {
            auto& v = self.harmonic_spectrum();
            size_t N = v.size();
            py::array_t<double> out({N, (size_t)2});
            auto o = out.mutable_unchecked<2>();
            for (size_t i = 0; i < N; ++i) {
                o(i, 0) = v[i].real();
                o(i, 1) = v[i].imag();
            }
            return out;
        })
        .def_property_readonly("latency_frames",  &HPSS::latency_frames)
        .def_property_readonly("latency_seconds", &HPSS::latency_seconds)
        .def_static("separate_signal",
            [](py::array_t<double> sig, int fft, int hop, int sr,
               int tk, int fk, double pw) {
                auto r = sig.unchecked<1>();
                RVec v(r.data(0), r.data(0) + r.size());
                auto [h, p] = HPSS::separate_signal(v, fft, hop, sr, tk, fk, pw);
                return py::make_tuple(
                    py::array_t<double>(h.size(), h.data()),
                    py::array_t<double>(p.size(), p.data()));
            },
            py::arg("signal"), py::arg("fft_size"), py::arg("hop_size"),
            py::arg("sr"),
            py::arg("time_kernel") = 17, py::arg("freq_kernel") = 17,
            py::arg("power") = 2.0);
}